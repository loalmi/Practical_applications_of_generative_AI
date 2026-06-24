"""
pipeline_fixed.py — RAG с фиксированным чанкингом (fixed-size)
Для сравнения стратегий чанкинга в ДЗ.
"""

import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from llm_client import get_model, make_client
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from schema import RAGAnswer

CHUNK_SIZE = 80

client = make_client()
MODEL = get_model()

DATA_DIR = Path(__file__).parent / "data"
BM25_CACHE = Path(__file__).parent / "bm25_cache_fixed.json"
FAISS_INDEX = "faiss_fixed.index"
CHUNKS_FILE = Path(__file__).parent / "chunks_fixed.json"

print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)


def tokenize_ru(text: str):
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text_fixed(text: str) -> list[str]:
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunks.append(text[i:i+CHUNK_SIZE])
    return chunks


def ingest():
    all_chunks = []
    all_ids = []

    files = sorted(DATA_DIR.glob("*.txt"))
    print(f"Найдено файлов: {len(files)}")
    print(f"Размер чанка (fixed-size): {CHUNK_SIZE} символов")

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text_fixed(text)
        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
        print(f"  {f.stem}: {len(chunks)} чанков")

    print("Вычисляю эмбеддинги...", flush=True)
    t0 = time.time()
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    print(f"Эмбеддинги готовы за {time.time()-t0:.1f}с", flush=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FAISS_INDEX))
    print(f"FAISS индекс сохранён: {FAISS_INDEX}")

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(
        json.dumps(bm25_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"BM25-кэш сохранён: {BM25_CACHE}")

    CHUNKS_FILE.write_text(
        json.dumps({"ids": all_ids, "texts": all_chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Чанки сохранены: {CHUNKS_FILE}")

    print(f"\nИндексировано: {len(all_ids)} чанков из {len(files)} файлов")


def load_faiss():
    index = faiss.read_index(str(FAISS_INDEX))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks["ids"], chunks["texts"]


def load_bm25():
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8-sig"))
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


def dense_search(query: str, k: int = 15):
    index, ids, texts = load_faiss()
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb.astype(np.float32), k)
    return [ids[i] for i in indices[0]], [texts[i] for i in indices[0]]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60):
    dense_ids, dense_texts = dense_search(query, top)

    bm25, bm25_ids, bm25_texts = load_bm25()
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]
    sparse_texts = [bm25_texts[i] for i in bm25_order]

    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank + 1)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense_ids):
        text_by_id[did] = dense_texts[i]
    for i, sid in enumerate(sparse_ids):
        text_by_id[sid] = sparse_texts[i]

    docs = [text_by_id[i] for i in top_ids]
    return {"ids": [top_ids], "documents": [docs]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по архиву документов.\n"
        "Опирайся ТОЛЬКО на контекст ниже.\n\n"
        "Правила:\n"
        "1. Если в контексте нет ответа — скажи 'В контексте нет информации'.\n"
        "2. Если есть ответ — приведи 1-5 точных цитат.\n"
        "3. В `sources` — id блоков.\n"
        "4. В `confidence` — честная оценка.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ (строго в формате JSON):"
    )


def ask(query: str):
    print(f"Поиск (fixed-size, dense + sparse + RRF): {query}")
    t0 = time.time()
    hits = hybrid_retrieve(query, k=5)
    found = hits["ids"][0]
    print(f"   нашёл {len(found)} чанков за {time.time()-t0:.1f}с")

    print("Генерация ответа...")
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"   ответ за {time.time()-t1:.1f}с")

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp)
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline_fixed.py {ingest|ask} [вопрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline_fixed.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
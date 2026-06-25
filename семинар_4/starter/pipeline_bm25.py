"""
pipeline_bm25.py — RAG только на BM25 (без ChromaDB) - не работает, возможно с windows
"""

import json
import re
import sys
import time
from pathlib import Path

from llm_client import get_model, make_client
from rank_bm25 import BM25Okapi
from schema import RAGAnswer

client = make_client()
MODEL = get_model()

DATA_DIR = Path(__file__).parent / "data"
BM25_CACHE = Path(__file__).parent / "bm25_cache.json"


def tokenize_ru(text: str):
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def ingest_bm25():
    """Индексация только для BM25 (без ChromaDB)"""
    all_chunks = []
    all_ids = []

    files = sorted(DATA_DIR.glob("*.txt"))
    print(f"Найдено файлов: {len(files)}")

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
        print(f"  {f.stem}: {len(chunks)} чанков")

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(
        json.dumps(bm25_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nИндексировано: {len(all_ids)} чанков из {len(files)} файлов")
    print(f"BM25-кэш сохранён в {BM25_CACHE.name}")


def load_bm25():
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8-sig"))
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


def search(query: str, k: int = 5):
    bm25, ids, texts = load_bm25()
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)
    ordered = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
    return [ids[i] for i in ordered], [texts[i] for i in ordered]


def build_prompt(query: str, docs: list[str], ids: list[str]) -> str:
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по архиву фокус-групп.\n"
        "Опирайся ТОЛЬКО на контекст ниже.\n\n"
        "Правила:\n"
        "1. Опирайся ТОЛЬКО на контекст ниже.\n"
        "2. В `quotes` — 1-5 точных цитат.\n"
        "3. В `sources` — id блоков.\n"
        "4. В `confidence` — честная оценка.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\nОтвет:"
    )


def ask(query: str):
    print(f"Поиск по BM25: {query}")
    t0 = time.time()
    ids, docs = search(query, k=5)
    print(f"   нашёл {len(docs)} чанков за {time.time()-t0:.1f}с")

    print("Генерация ответа...")
    t1 = time.time()
    prompt = build_prompt(query, docs, ids)
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
    for i in ids:
        print(f"  {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline_bm25.py {ingest|ask} [вопрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest_bm25()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline_bm25.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
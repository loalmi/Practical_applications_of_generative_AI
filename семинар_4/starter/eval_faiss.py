"""
eval_faiss.py — Оценка гибридного RAG на FAISS + BM25 + RRF
"""

import argparse
import json
from pathlib import Path

from pipeline_faiss import hybrid_retrieve
from pipeline_bm25 import search as bm25_search

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    """Доля gold_sources, попавших в ТОП-K чанков."""
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources) if gold_sources else 0.0


def run(method: str = "hybrid", k: int = 5, verbose: bool = True) -> dict:
    gold = load_gold()
    total = 0.0
    results = []

    label_map = {
        "hybrid": "HYBRID (DENSE + BM25 + RRF)",
        "bm25": "BM25 ONLY",
    }
    print(f"\n==={label_map.get(method, method)}===\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        if method == "hybrid":
            hits = hybrid_retrieve(q, k=k)
            retrieved_ids = hits["ids"][0]
        else:  # bm25
            retrieved_ids, _ = bm25_search(q, k=k)

        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        results.append({
            "id": item["id"],
            "type": item["type"],
            "score": score,
            "gold": gold_sources,
            "retrieved_sources": retrieved_sources,
        })

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(f"  [{item['id']:2d}] {item['type']:25s}  "
                  f"hit@{k} = {score:.2f}  {mark}  {q}")

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return {"mean": mean, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["hybrid", "bm25"], default="hybrid",
                        help="Метод поиска: hybrid (FAISS+BM25+RRF) или bm25")
    parser.add_argument("--k", type=int, default=5,
                        help="Количество результатов для оценки")
    parser.add_argument("--quiet", action="store_true",
                        help="Скрыть подробный вывод")
    args = parser.parse_args()

    run(method=args.method, k=args.k, verbose=not args.quiet)


if __name__ == "__main__":
    main()
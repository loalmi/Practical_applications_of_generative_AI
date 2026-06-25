"""
eval.py — Универсальная оценка RAG с сохранением результатов
Поддерживает любой модуль: pipeline (recursive) или pipeline_fixed (fixed-size)
"""

import argparse
import importlib
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"
RESULTS_DIR = Path(__file__).parent / "eval_results"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8-sig"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources) if gold_sources else 0.0


def run(hybrid_retrieve, k: int = 5, strategy: str = "hybrid", verbose: bool = True) -> dict:
    gold = load_gold()
    total = 0.0
    results = []

    print(f"\n=== {strategy.upper()} | chunking strategy ===\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        hits = hybrid_retrieve(q, k=k)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": q,
            "score": score,
            "gold_sources": ", ".join(gold_sources),
            "retrieved_sources": ", ".join(retrieved_sources),
            "hit": score == 1.0,
        })

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(f"  [{item['id']:2d}] {item['type']:25s}  "
                  f"hit@{k} = {score:.2f}  {mark}  {q[:50]}...")

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")

    return {"mean": mean, "results": results}


def save_results(results: dict, strategy: str, k: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{strategy}_k{k}_{timestamp}"

    json_path = RESULTS_DIR / f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": strategy,
            "k": k,
            "hit_rate": results["mean"],
            "timestamp": timestamp,
            "results": results["results"],
        }, f, ensure_ascii=False, indent=2)
    print(f"  JSON сохранён: {json_path}")

    csv_path = RESULTS_DIR / f"{prefix}.csv"
    df = pd.DataFrame(results["results"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV сохранён: {csv_path}")

    summary_path = RESULTS_DIR / "summary.csv"
    summary_data = {
        "strategy": strategy,
        "k": k,
        "hit_rate": results["mean"],
        "timestamp": timestamp,
    }
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_data])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  Сводка обновлена: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Универсальная оценка RAG")
    parser.add_argument("--k", type=int, default=5,
                        help="Количество результатов для оценки")
    parser.add_argument("--strategy", type=str, default="hybrid",
                        help="Название стратегии (например: fixed_2000, recursive_400)")
    parser.add_argument("--module", type=str, default="pipeline",
                        help="Имя модуля: pipeline (recursive) или pipeline_fixed (fixed-size)")
    parser.add_argument("--quiet", action="store_true",
                        help="Скрыть подробный вывод")
    parser.add_argument("--no-save", action="store_true",
                        help="Не сохранять результаты")
    args = parser.parse_args()

    try:
        module = importlib.import_module(args.module)
        hybrid_retrieve = module.hybrid_retrieve
    except ImportError:
        print(f"Ошибка: модуль '{args.module}' не найден.")
        print("Доступные модули: pipeline (recursive), pipeline_fixed (fixed-size)")
        return
    except AttributeError:
        print(f"Ошибка: в модуле '{args.module}' нет функции hybrid_retrieve.")
        return

    results = run(hybrid_retrieve, k=args.k, strategy=args.strategy, verbose=not args.quiet)

    if not args.no_save:
        save_results(results, args.strategy, args.k)


if __name__ == "__main__":
    main()
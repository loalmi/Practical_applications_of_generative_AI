"""
eval_val.py — запуск eval с валидатором
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pwc import run_case, CASES
from orchestrator import run_pwc
import orchestrator

# Сохраняем оригинальную функцию
original_run_pwc = orchestrator.run_pwc

# Подменяем на версию с validate=True
def run_pwc_with_validate(*args, **kwargs):
    kwargs["validate"] = True
    return original_run_pwc(*args, **kwargs)

orchestrator.run_pwc = run_pwc_with_validate

def main():
    print("=" * 60)
    print("EVAL С ВАЛИДАТОРОМ (PWC + VALIDATE)")
    print("=" * 60)
    
    results = []
    for case in CASES:
        print(f"\n=== {case['id']}: {case['query'][:60]}...")
        result = run_case(case, n=5)  # один прогон для скорости
        results.append(result)
        print(f"   single: {result['single']['pass']}/1    pwc: {result['pwc']['pass']}/1")
    
    print("\n" + "=" * 60)
    print("ИТОГО (с валидатором):")
    for r in results:
        print(f"  {r['id']}: single {r['single']['pass']}/1  pwc {r['pwc']['pass']}/1  — {r['query'][:60]}...")
    
    # Сохраняем результаты
    import json
    with open("eval_val_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nРезультаты сохранены: eval_val_results.json")

if __name__ == "__main__":
    main()
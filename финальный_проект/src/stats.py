"""
stats.py — статистика по eval_results.json
"""

import json
from collections import Counter

print("=" * 60)
print("СТАТИСТИКА EVAL")
print("=" * 60)

with open("output/eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Всего кейсов: {data['total_cases']}")
print(f"Всего LLM-вызовов: {data['total_llm_calls']}")
print(f"Всего токенов: {data['total_tokens']}")
print(f"Стоимость: ${data['estimated_cost']}")
print()

pass_count = sum(1 for r in data["results"] if r["pass_fail"] == "PASS")
fail_count = len(data["results"]) - pass_count
print(f"PASS: {pass_count}")
print(f"FAIL: {fail_count}")
print()

scores = [r["judge_score"] for r in data["results"]]
avg_score = sum(scores) / len(scores) if scores else 0
print(f"Средняя оценка судьи: {avg_score:.2f}/5")
print()

score_dist = Counter(scores)
print("Распределение оценок:")
for score in sorted(score_dist.keys()):
    pct = score_dist[score] / len(scores) * 100
    print(f"  {score}: {score_dist[score]} ({pct:.1f}%)")
print()

valid_options = sum(1 for r in data["results"] if r["valid_option"])
matches_profile = sum(1 for r in data["results"] if r["matches_profile"])
print(f"Допустимые варианты: {valid_options}/{len(data['results'])} ({valid_options/len(data['results'])*100:.1f}%)")
print(f"Соответствие профилю: {matches_profile}/{len(data['results'])} ({matches_profile/len(data['results'])*100:.1f}%)")
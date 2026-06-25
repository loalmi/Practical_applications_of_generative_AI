"""
hallucination_report.py — отчёт о галлюцинациях
"""

import json
import os
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Загружаем результаты
with open(f"{OUTPUT_DIR}/survey_results_validated.json", "r", encoding="utf-8") as f:
    results = json.load(f)

from questions import SURVEY_QUESTIONS

# Собираем допустимые варианты
valid_options = {}
for q in SURVEY_QUESTIONS:
    valid_options[q["id"]] = q["options"]

hallucinations = {
    "total_responses": 0,
    "invalid_options": [],
    "profile_contradictions": [],
    "hallucinated_categories": Counter(),
    "examples": []
}

for respondent in results:
    profile = respondent["demographics"]
    for ans in respondent["answers"]:
        q_id = ans["question_id"]
        answer = ans["answer"]
        hallucinations["total_responses"] += 1

        # 1. Проверка на допустимые варианты
        if answer not in valid_options.get(q_id, []):
            hallucinations["invalid_options"].append({
                "question": q_id,
                "answer": answer,
                "profile": profile
            })
            hallucinations["hallucinated_categories"][answer] += 1

        # 2. Проверка на противоречия профилю
        # (логика: если возраст < 25 и ответ "Полностью удовлетворен" — противоречие)
        if profile["age"] < 25 and answer == "Полностью удовлетворен":
            hallucinations["profile_contradictions"].append({
                "question": q_id,
                "answer": answer,
                "profile": profile,
                "reason": "Молодой возраст + максимальная удовлетворённость"
            })

# Берём 3 примера
hallucinations["examples"] = (
    hallucinations["invalid_options"][:2] + 
    hallucinations["profile_contradictions"][:1]
)

# Сохраняем
with open(f"{OUTPUT_DIR}/hallucination_report.json", "w", encoding="utf-8") as f:
    json.dump(hallucinations, f, ensure_ascii=False, indent=2)

print(f"Сохранено: {OUTPUT_DIR}/hallucination_report.json")
print(f"Всего ответов: {hallucinations['total_responses']}")
print(f"Невалидных ответов: {len(hallucinations['invalid_options'])}")
print(f"Противоречий профилю: {len(hallucinations['profile_contradictions'])}")
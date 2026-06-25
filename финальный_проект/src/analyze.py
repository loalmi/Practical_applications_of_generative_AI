"""
analyze.py — анализ результатов опроса
"""

import os
import json
import pandas as pd
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Загружаем результаты
# Для первого опроса:
# with open(f"{OUTPUT_DIR}/survey_results.json", "r", encoding="utf-8") as f:
#     results = json.load(f)

# Для второго опроса:
with open(f"{OUTPUT_DIR}/survey_results_validated.json", "r", encoding="utf-8") as f:
    results = json.load(f)



# Загружаем реальные данные РМЭЗ
df_real = pd.read_csv(f"{OUTPUT_DIR}/rlms_clean.csv")

print("=" * 60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ ОПРОСА")
print("=" * 60)

# --- 1. Распределение ответов по вопросам ---
print("\n1. РАСПРЕДЕЛЕНИЕ ОТВЕТОВ ПО ВОПРОСАМ")
print("-" * 60)

question_answers = {}
for respondent in results:
    for ans in respondent["answers"]:
        q_id = ans["question_id"]
        q_text = ans["question"]
        answer = ans["answer"]
        if q_id not in question_answers:
            question_answers[q_id] = {"text": q_text, "answers": []}
        question_answers[q_id]["answers"].append(answer)

for q_id, data in question_answers.items():
    counter = Counter(data["answers"])
    total = sum(counter.values())
    print(f"\n{q_id}: {data['text'][:60]}...")
    for option, count in counter.most_common():
        pct = count / total * 100
        print(f"  {option}: {count} ({pct:.1f}%)")

# --- 2. Сравнение по группам (пол) ---
print("\n2. СРАВНЕНИЕ ПО ГРУППАМ (ПОЛ)")
print("-" * 60)

gender_groups = {"Мужской": [], "Женский": []}

for respondent in results:
    gender = respondent["demographics"]["gender"]
    if gender in gender_groups:
        for ans in respondent["answers"]:
            gender_groups[gender].append({
                "question_id": ans["question_id"],
                "answer": ans["answer"]
            })

for q_id, data in question_answers.items():
    print(f"\n{q_id}: {data['text'][:50]}...")
    for gender in ["Мужской", "Женский"]:
        answers = [a["answer"] for a in gender_groups[gender] if a["question_id"] == q_id]
        if answers:
            counter = Counter(answers)
            total = len(answers)
            most_common = counter.most_common(1)[0]
            print(f"  {gender}: {most_common[0]} ({most_common[1]}/{total})")

# --- 3. Сравнение по возрастным группам ---
print("\n3. СРАВНЕНИЕ ПО ВОЗРАСТНЫМ ГРУППАМ")
print("-" * 60)

def get_age_group(age):
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55+"

age_groups = {}
for respondent in results:
    age = respondent["demographics"]["age"]
    group = get_age_group(age)
    if group not in age_groups:
        age_groups[group] = []
    for ans in respondent["answers"]:
        age_groups[group].append({
            "question_id": ans["question_id"],
            "answer": ans["answer"]
        })

for q_id, data in question_answers.items():
    print(f"\n{q_id}: {data['text'][:50]}...")
    for group in sorted(age_groups.keys()):
        answers = [a["answer"] for a in age_groups[group] if a["question_id"] == q_id]
        if answers:
            counter = Counter(answers)
            total = len(answers)
            most_common = counter.most_common(1)[0]
            print(f"  {group}: {most_common[0]} ({most_common[1]}/{total})")

# --- 4. Поиск двусмысленных вопросов ---
print("\n4. ПОИСК ДВУСМЫСЛЕННЫХ ВОПРОСОВ")
print("-" * 60)

for q_id, data in question_answers.items():
    counter = Counter(data["answers"])
    total = sum(counter.values())
    if len(counter) > 2:
        max_pct = max(counter.values()) / total * 100
        if max_pct < 60:
            print(f"{q_id}: {data['text'][:60]}...")
            print(f"   Разброс: {len(counter)} вариантов, максимум {max_pct:.1f}%")
            print(f"   Возможно, вопрос требует уточнения")

# --- 5. Сравнение с реальными данными РМЭЗ ---
print("\n5. СРАВНЕНИЕ С РЕАЛЬНЫМИ ДАННЫМИ РМЭЗ")
print("-" * 60)

# Сопоставляем Q1 (удовлетворённость материальным положением) с ccj65
try:
    real_counts = df_real['ccj65'].value_counts().sort_index()
    print("Реальные данные (ccj65 — удовлетворённость жизнью):")
    for val, count in real_counts.items():
        if pd.notna(val):
            print(f"  {val}: {count}")

    # Сравниваем с синтетическими ответами
    q1_answers = []
    for respondent in results:
        for ans in respondent["answers"]:
            if ans["question_id"] == "Q1":
                q1_answers.append(ans["answer"])
    
    if q1_answers:
        synth_counts = Counter(q1_answers)
        print("\nСинтетические ответы (Q1):")
        for opt, count in synth_counts.most_common():
            print(f"  {opt}: {count}")
except Exception as e:
    print(f"Не удалось сравнить с реальными данными: {e}")
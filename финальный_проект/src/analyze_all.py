"""
analyze_all.py — полный анализ и визуализация
Сравнивает: без валидатора / с валидатором / реальные данные РМЭЗ
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Создаём папку для графиков
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Настройка стиля
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(filename):
    """Загружает результаты опроса"""
    try:
        with open(f"{OUTPUT_DIR}/{filename}", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_question_answers(results):
    """Извлекает ответы по вопросам"""
    qa = {}
    for respondent in results:
        for ans in respondent["answers"]:
            q_id = ans["question_id"]
            q_text = ans["question"]
            answer = ans["answer"]
            if q_id not in qa:
                qa[q_id] = {"text": q_text, "answers": []}
            qa[q_id]["answers"].append(answer)
    return qa


def get_q1_real_distribution(df):
    """Получает распределение Q1 из реальных данных РМЭЗ"""
    # ccj65: 1=Полностью удовлетворен, 2=Скорее, 3=И да и нет, 4=Не очень, 5=Совсем нет
    mapping = {
        1: "Полностью удовлетворен",
        2: "Скорее удовлетворен",
        3: "И да, и нет",
        4: "Не очень удовлетворен",
        5: "Совсем не удовлетворен"
    }
    real_counts = df['ccj65'].value_counts().sort_index()
    result = {}
    for val, count in real_counts.items():
        if pd.notna(val) and val in mapping:
            result[mapping[val]] = count
    return result


print("=" * 60)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ТРЁХ ИСТОЧНИКОВ")
print("=" * 60)

# 1. Загружаем данные
results_no_val = load_results("survey_results.json")
results_with_val = load_results("survey_results_validated.json")
df_real = pd.read_csv(f"{OUTPUT_DIR}/rlms_clean.csv")

print(f"Без валидатора: {len(results_no_val) if results_no_val else 0} респондентов")
print(f"С валидатором: {len(results_with_val) if results_with_val else 0} респондентов")
print(f"Реальные данные: {len(df_real)} записей")

# 2. Извлекаем ответы
qa_no_val = get_question_answers(results_no_val) if results_no_val else {}
qa_with_val = get_question_answers(results_with_val) if results_with_val else {}
real_q1 = get_q1_real_distribution(df_real)

print("\n" + "=" * 60)
print("РАСПРЕДЕЛЕНИЕ ОТВЕТОВ НА ВОПРОСЫ")
print("=" * 60)

# 3. Сравнение по каждому вопросу
for q_id in qa_with_val.keys():
    print(f"\n{q_id}: {qa_with_val[q_id]['text'][:60]}...")
    
    no_counter = Counter(qa_no_val.get(q_id, {}).get("answers", []))
    with_counter = Counter(qa_with_val.get(q_id, {}).get("answers", []))
    
    no_total = sum(no_counter.values()) or 1
    with_total = sum(with_counter.values()) or 1
    
    all_opts = set(no_counter.keys()) | set(with_counter.keys())
    
    print(f"  {'Вариант':<35} | {'Без валид.':<12} | {'С валид.':<12} | Δ")
    print("  " + "-" * 80)
    
    for opt in sorted(all_opts):
        no_pct = no_counter.get(opt, 0) / no_total * 100
        with_pct = with_counter.get(opt, 0) / with_total * 100
        diff = with_pct - no_pct
        print(f"  {opt:<35} | {no_pct:>5.1f}% ({no_counter.get(opt, 0):>2}) | {with_pct:>5.1f}% ({with_counter.get(opt, 0):>2}) | {diff:+5.1f}%")

# 4. Сравнение с реальными данными (Q1)
print("\n" + "=" * 60)
print("СРАВНЕНИЕ Q1 С РЕАЛЬНЫМИ ДАННЫМИ РМЭЗ")
print("=" * 60)

if real_q1 and qa_with_val.get("Q1"):
    print(f"\n{'Вариант':<35} | {'Реальные данные':<15} | {'Синтетика':<12}")
    print("  " + "-" * 70)
    
    # Нормализуем синтетику
    synth_counter = Counter(qa_with_val["Q1"]["answers"])
    synth_total = sum(synth_counter.values())
    
    all_q1_opts = set(real_q1.keys()) | set(synth_counter.keys())
    for opt in sorted(all_q1_opts):
        real_pct = real_q1.get(opt, 0) / sum(real_q1.values()) * 100
        synth_pct = synth_counter.get(opt, 0) / synth_total * 100
        print(f"  {opt:<35} | {real_pct:>5.1f}% ({real_q1.get(opt, 0):>5}) | {synth_pct:>5.1f}%")

# 5. Визуализация
print("\n" + "=" * 60)
print("СОЗДАНИЕ ГРАФИКОВ")
print("=" * 60)

# График 1: Сравнение Q1 (все три источника)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Без валидатора
if qa_no_val.get("Q1"):
    counter = Counter(qa_no_val["Q1"]["answers"])
    axes[0].bar(counter.keys(), counter.values())
    axes[0].set_title("Без валидатора (Q1)")
    axes[0].tick_params(axis='x', rotation=45)

# С валидатором
if qa_with_val.get("Q1"):
    counter = Counter(qa_with_val["Q1"]["answers"])
    axes[1].bar(counter.keys(), counter.values())
    axes[1].set_title("С валидатором (Q1)")
    axes[1].tick_params(axis='x', rotation=45)

# Реальные данные
if real_q1:
    axes[2].bar(real_q1.keys(), real_q1.values())
    axes[2].set_title("Реальные данные РМЭЗ (Q1)")
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/q1_comparison.png", dpi=150)
print(f"График сохранён: {PLOTS_DIR}/q1_comparison.png")

# График 2: Сравнение всех вопросов (с валидатором)
if qa_with_val:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (q_id, data) in enumerate(qa_with_val.items()):
        if idx < 6:
            counter = Counter(data["answers"])
            axes[idx].bar(counter.keys(), counter.values())
            axes[idx].set_title(f"{q_id}")
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/all_questions_distribution.png", dpi=150)
    print(f"График сохранён: {PLOTS_DIR}/all_questions_distribution.png")

# График 3: Средняя уверенность (если есть)
if results_with_val:
    confidences = []
    for respondent in results_with_val:
        for ans in respondent["answers"]:
            if "confidence" in ans:
                confidences.append(ans["confidence"])
    
    if confidences:
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=10, edgecolor='black')
        plt.xlabel("Уверенность")
        plt.ylabel("Количество")
        plt.title("Распределение уверенности ответов (с валидатором)")
        plt.savefig(f"{PLOTS_DIR}/confidence_distribution.png", dpi=150)
        print(f"График сохранён: {PLOTS_DIR}/confidence_distribution.png")
        print(f"   Средняя уверенность: {np.mean(confidences):.2f}")

print("\n" + "=" * 60)

# --- 6. Сохранение таблиц ---
print("\n6. СОХРАНЕНИЕ ТАБЛИЦ")
print("-" * 60)

# Создаём папку для таблиц
TABLES_DIR = f"{OUTPUT_DIR}/tables"
os.makedirs(TABLES_DIR, exist_ok=True)

# Таблица 1: Сравнение без валидатора vs с валидатором
table1_data = []
for q_id in qa_with_val.keys():
    no_counter = Counter(qa_no_val.get(q_id, {}).get("answers", []))
    with_counter = Counter(qa_with_val.get(q_id, {}).get("answers", []))
    no_total = sum(no_counter.values()) or 1
    with_total = sum(with_counter.values()) or 1
    
    all_opts = set(no_counter.keys()) | set(with_counter.keys())
    for opt in sorted(all_opts):
        table1_data.append({
            "Вопрос": q_id,
            "Вариант": opt,
            "Без валидатора (%)": round(no_counter.get(opt, 0) / no_total * 100, 1),
            "Без валидатора (N)": no_counter.get(opt, 0),
            "С валидатором (%)": round(with_counter.get(opt, 0) / with_total * 100, 1),
            "С валидатором (N)": with_counter.get(opt, 0),
            "Изменение (п.п.)": round(with_counter.get(opt, 0) / with_total * 100 - no_counter.get(opt, 0) / no_total * 100, 1)
        })

df_table1 = pd.DataFrame(table1_data)
df_table1.to_csv(f"{TABLES_DIR}/comparison_no_val_vs_with_val.csv", index=False, encoding="utf-8-sig")
print(f"Таблица 1 сохранена: {TABLES_DIR}/comparison_no_val_vs_with_val.csv")

# Таблица 2: Сравнение с реальными данными РМЭЗ (Q1)
if real_q1 and qa_with_val.get("Q1"):
    table2_data = []
    synth_counter = Counter(qa_with_val["Q1"]["answers"])
    synth_total = sum(synth_counter.values())
    real_total = sum(real_q1.values())
    
    all_q1_opts = set(real_q1.keys()) | set(synth_counter.keys())
    for opt in sorted(all_q1_opts):
        table2_data.append({
            "Вариант": opt,
            "Реальные данные (%)": round(real_q1.get(opt, 0) / real_total * 100, 1),
            "Реальные данные (N)": real_q1.get(opt, 0),
            "Синтетика (%)": round(synth_counter.get(opt, 0) / synth_total * 100, 1),
            "Синтетика (N)": synth_counter.get(opt, 0),
            "Разница (п.п.)": round(synth_counter.get(opt, 0) / synth_total * 100 - real_q1.get(opt, 0) / real_total * 100, 1)
        })
    
    df_table2 = pd.DataFrame(table2_data)
    df_table2.to_csv(f"{TABLES_DIR}/comparison_real_vs_synthetic.csv", index=False, encoding="utf-8-sig")
    print(f"Таблица 2 сохранена: {TABLES_DIR}/comparison_real_vs_synthetic.csv")

# Таблица 3: Статистика по вопросам (с валидатором)
if qa_with_val:
    table3_data = []
    for q_id, data in qa_with_val.items():
        counter = Counter(data["answers"])
        total = sum(counter.values())
        table3_data.append({
            "Вопрос": q_id,
            "Текст вопроса": data["text"][:80],
            "Всего ответов": total,
            "Уникальных ответов": len(counter),
            "Самый частый ответ": counter.most_common(1)[0][0] if counter else "-",
            "Доля самого частого (%)": round(counter.most_common(1)[0][1] / total * 100, 1) if counter else 0,
            "Количество вариантов": len(counter)
        })
    
    df_table3 = pd.DataFrame(table3_data)
    df_table3.to_csv(f"{TABLES_DIR}/question_statistics.csv", index=False, encoding="utf-8-sig")
    print(f"Таблица 3 сохранена: {TABLES_DIR}/question_statistics.csv")

# Таблица 4: Сравнение по полу (с валидатором)
if qa_with_val and results_with_val:
    table4_data = []
    gender_groups = {"Мужской": [], "Женский": []}
    
    for respondent in results_with_val:
        gender = respondent["demographics"]["gender"]
        if gender in gender_groups:
            for ans in respondent["answers"]:
                gender_groups[gender].append({
                    "question_id": ans["question_id"],
                    "answer": ans["answer"]
                })
    
    for q_id in qa_with_val.keys():
        for gender in ["Мужской", "Женский"]:
            answers = [a["answer"] for a in gender_groups[gender] if a["question_id"] == q_id]
            if answers:
                counter = Counter(answers)
                total = len(answers)
                most_common = counter.most_common(1)[0]
                table4_data.append({
                    "Вопрос": q_id,
                    "Пол": gender,
                    "Всего ответов": total,
                    "Самый частый ответ": most_common[0],
                    "Доля самого частого (%)": round(most_common[1] / total * 100, 1)
                })
    
    df_table4 = pd.DataFrame(table4_data)
    df_table4.to_csv(f"{TABLES_DIR}/comparison_by_gender.csv", index=False, encoding="utf-8-sig")
    print(f"Таблица 4 сохранена: {TABLES_DIR}/comparison_by_gender.csv")

print(f"\nВсе таблицы сохранены в: {TABLES_DIR}")


print("АНАЛИЗ ЗАВЕРШЁН")
print(f"Графики сохранены в: {PLOTS_DIR}")
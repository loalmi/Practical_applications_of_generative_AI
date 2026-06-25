"""
analysis.py — расширенный анализ заявок на курсы ДПО.
Аналог 5_analysis.py из семинара.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Загружаем данные
with open("applications.json", "r", encoding="utf-8") as f:
    applications = json.load(f)

df = pd.DataFrame(applications)

# Распаковываем address
df['city'] = df['address'].apply(lambda x: x['city'])
df['district'] = df['address'].apply(lambda x: x['district'])
df = df.drop('address', axis=1)

print(f"Загружено: {len(df)} заявок")
print("-" * 50)

# ─── 1. Распределение по городам ───
city_counts = df['city'].value_counts()
print("\nТоп-города:")
for city, count in city_counts.items():
    pct = count / len(df) * 100
    print(f"  {city}: {count} ({pct:.1f}%)")

# ─── 2. Распределение по специальностям ───
spec_counts = df['speciality'].value_counts()
print("\nТоп-специальности:")
for spec, count in spec_counts.items():
    pct = count / len(df) * 100
    print(f"  {spec}: {count} ({pct:.1f}%)")

# ─── 3. Распределение по курсам ───
course_counts = df['desired_course'].value_counts()
print("\nТоп-курсы:")
for course, count in course_counts.items():
    pct = count / len(df) * 100
    print(f"  {course}: {count} ({pct:.1f}%)")

# ─── 4. Кросс-таблица: город × специальность ───
cross_tab = pd.crosstab(df['city'], df['speciality'])
print("\nКросс-таблица: город × специальность")
print(cross_tab)

# Сохраняем кросс-таблицу в CSV
cross_tab.to_csv("cross_tab.csv")
print("\nКросс-таблица сохранена в cross_tab.csv")

# ─── 5. Медианный возраст по специальностям ───
median_age = df.groupby('speciality')['age'].median()
print("\nМедианный возраст по специальностям:")
for spec, age in median_age.items():
    print(f"  {spec}: {age:.0f} лет")

# ─── 6. Медианный опыт по специальностям ───
median_exp = df.groupby('speciality')['years_of_experience'].median()
print("\nМедианный опыт по специальностям:")
for spec, exp in median_exp.items():
    print(f"  {spec}: {exp:.0f} лет")

# ─── 7. Поиск нереалистичных комбинаций ───
print("\nНереалистичные комбинации:")

# Комбинация 1: Молодой с большим опытом
suspect1 = df[(df['age'] < 25) & (df['years_of_experience'] > 5)]
if not suspect1.empty:
    print(f"  - Молодой (<25 лет) с большим опытом (>5 лет): {len(suspect1)} записей")
    for _, row in suspect1.iterrows():
        print(f"    {row['full_name']}: {row['age']} лет, опыт {row['years_of_experience']} лет")

# Комбинация 2: Пожилой с малым опытом
suspect2 = df[(df['age'] > 55) & (df['years_of_experience'] < 5)]
if not suspect2.empty:
    print(f"  - Пожилой (>55 лет) с малым опытом (<5 лет): {len(suspect2)} записей")
    for _, row in suspect2.iterrows():
        print(f"    {row['full_name']}: {row['age']} лет, опыт {row['years_of_experience']} лет")

# Комбинация 3: Возраст и год окончания (проверка валидатора)
df['expected_graduation'] = 2026 - df['age'] + 22
df['graduation_diff'] = abs(df['graduation_year'] - df['expected_graduation'])
suspect3 = df[df['graduation_diff'] > 5]
if not suspect3.empty:
    print(f"  - Год окончания не соответствует возрасту: {len(suspect3)} записей")
    for _, row in suspect3.iterrows():
        print(f"    {row['full_name']}: {row['age']} лет, год окончания {row['graduation_year']}")

# ─── 8. Сохраняем отчёт ───
with open("report.md", "w", encoding="utf-8") as f:
    f.write("# Отчёт по генерации заявок на курсы ДПО\n\n")
    
    f.write("## Города\n\n")
    for city, count in city_counts.items():
        pct = count / len(df) * 100
        f.write(f"- {city}: {count} ({pct:.1f}%)\n")
    
    f.write("\n## Специальности\n\n")
    for spec, count in spec_counts.items():
        pct = count / len(df) * 100
        f.write(f"- {spec}: {count} ({pct:.1f}%)\n")
    
    f.write("\n## Курсы\n\n")
    for course, count in course_counts.items():
        pct = count / len(df) * 100
        f.write(f"- {course}: {count} ({pct:.1f}%)\n")
    
    f.write("\n## Кросс-таблица: город × специальность\n\n")
    f.write(cross_tab.to_markdown())
    
    f.write("\n\n## Медианный возраст по специальностям\n\n")
    for spec, age in median_age.items():
        f.write(f"- {spec}: {age:.0f} лет\n")
    
    f.write("\n## Медианный опыт по специальностям\n\n")
    for spec, exp in median_exp.items():
        f.write(f"- {spec}: {exp:.0f} лет\n")

print("\nОтчёт сохранён в report.md")

# ─── 9. Строим графики ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Города
ax1 = axes[0, 0]
city_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Распределение по городам', fontsize=12)
ax1.set_xlabel('Город')
ax1.set_ylabel('Количество')
ax1.tick_params(axis='x', rotation=45)

# График 2: Специальности
ax2 = axes[0, 1]
spec_counts.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Распределение по специальностям', fontsize=12)
ax2.set_xlabel('Специальность')
ax2.set_ylabel('Количество')
ax2.tick_params(axis='x', rotation=45)

# График 3: Курсы
ax3 = axes[1, 0]
course_counts.plot(kind='bar', ax=ax3, color='salmon')
ax3.set_title('Распределение по курсам', fontsize=12)
ax3.set_xlabel('Курс')
ax3.set_ylabel('Количество')
ax3.tick_params(axis='x', rotation=45)

# График 4: Возраст по специальностям
ax4 = axes[1, 1]
df.boxplot(column='age', by='speciality', ax=ax4)
ax4.set_title('Возраст по специальностям', fontsize=12)
ax4.set_xlabel('Специальность')
ax4.set_ylabel('Возраст')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('analysis_plots.png')
print("analysis_plots.png сохранён")

print("\nАнализ завершён!")
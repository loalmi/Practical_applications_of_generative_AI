"""
analyze.py — анализ распределения заявок.
Строит графики для cities.png и specialities.png.
"""

import json
import matplotlib.pyplot as plt
from collections import Counter

# Загружаем данные
with open("applications.json", "r", encoding="utf-8") as f:
    applications = json.load(f)

n = len(applications)
print(f"Загружено: {n} заявок")

# ─── Распределение по городам ───
cities = [app["address"]["city"] for app in applications]
city_counts = Counter(cities)

print("\nРаспределение по городам:")
for city, count in city_counts.most_common():
    pct = count / n * 100
    print(f"  {city}: {count} ({pct:.1f}%)")
    if pct > 40:
        print(f"    ПРЕВЫШЕН ПОРОГ 40%!")

# ─── Распределение по специальностям ───
specialities = [app["speciality"] for app in applications]
spec_counts = Counter(specialities)

print("\nРаспределение по специальностям:")
for spec, count in spec_counts.most_common():
    pct = count / n * 100
    print(f"  {spec}: {count} ({pct:.1f}%)")
    if pct > 35:
        print(f"    ПРЕВЫШЕН ПОРОГ 35%!")

# ─── Построение графиков ───
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График по городам
ax1 = axes[0]
bars1 = ax1.bar(city_counts.keys(), city_counts.values(), color="skyblue")
ax1.set_title("Распределение заявок по городам", fontsize=14)
ax1.set_xlabel("Город")
ax1.set_ylabel("Количество заявок")
ax1.tick_params(axis="x", rotation=45)
# Добавляем порог 40%
ax1.axhline(y=n * 0.4, color="red", linestyle="--", label="Порог 40%")
ax1.legend()

# График по специальностям
ax2 = axes[1]
bars2 = ax2.bar(spec_counts.keys(), spec_counts.values(), color="lightgreen")
ax2.set_title("Распределение по специальностям", fontsize=14)
ax2.set_xlabel("Специальность")
ax2.set_ylabel("Количество заявок")
ax2.tick_params(axis="x", rotation=45)
# Добавляем порог 35%
ax2.axhline(y=n * 0.35, color="red", linestyle="--", label="Порог 35%")
ax2.legend()

plt.tight_layout()
plt.savefig("cities.png")
print("\ncities.png сохранён")

plt.savefig("specialities.png")
print("specialities.png сохранён")

plt.show()
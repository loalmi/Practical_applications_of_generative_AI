"""
explore_data.py — просмотр структуры данных РМЭЗ
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Путь к данным
DATA_PATH = os.getenv("DATA_PATH", "input/r33h_os_83.dta")

print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ РМЭЗ")
print("=" * 60)

# Загружаем данные
df = pd.read_stata(DATA_PATH, convert_categoricals=False)

print(f"✅ Загружено: {len(df)} строк, {len(df.columns)} столбцов")
print()

# 1. Первые 5 строк
print("=" * 60)
print("ПЕРВЫЕ 5 СТРОК")
print("=" * 60)
print(df.head())
print()

# 2. Список всех переменных (первые 30)
print("=" * 60)
print("СПИСОК ПЕРЕМЕННЫХ (первые 30 из {})".format(len(df.columns)))
print("=" * 60)
for i, col in enumerate(df.columns[:30]):
    print(f"  {i+1:2d}. {col}")
print()

# 3. Типы данных
print("=" * 60)
print("ТИПЫ ДАННЫХ (первые 20)")
print("=" * 60)
print(df.dtypes.head(20))
print()

# 4. Основные статистики для числовых переменных
print("=" * 60)
print("СТАТИСТИКИ ПО ЧИСЛОВЫМ ПЕРЕМЕННЫМ")
print("=" * 60)
print(df.describe())
print()

# 5. Количество пропусков
print("=" * 60)
print("ПРОПУСКИ (первые 20 переменных)")
print("=" * 60)
print(df.isnull().sum().head(20))
print()

# 6. Поиск демографических переменных
print("=" * 60)
print("ПОИСК ДЕМОГРАФИЧЕСКИХ ПЕРЕМЕННЫХ")
print("=" * 60)
# demo_keywords = ["age", "sex", "gender", "edu", "educ", "school", "work", "job", "income", "family", "household"]
# Вместо английских ключевых слов, попробуйте русские:
demo_keywords = ["возраст", "пол", "образов", "доход", "занят", "работ", "семья", "регион", "город", "насел"]
found = []
for col in df.columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in demo_keywords):
        found.append(col)

print(f"Найдено {len(found)} потенциальных демографических переменных:")
for col in found[:20]:
    print(f"  - {col}")
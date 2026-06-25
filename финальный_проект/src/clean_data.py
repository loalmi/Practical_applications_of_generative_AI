"""
clean_data.py — очистка и подготовка данных РМЭЗ
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "input/r33iall_84.dta")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Специальные коды пропусков в РМЭЗ
MISSING_CODES = [
    99999997,  # ЗАТРУДНЯЮСЬ ОТВЕТИТЬ
    99999998,  # ОТКАЗ ОТ ОТВЕТА
    99999999,  # НЕТ ОТВЕТА
    99999996,  # НЕ ОТНОСИТСЯ / ДРУГОЕ
]

print("=" * 60)
print("ОЧИСТКА ДАННЫХ РМЭЗ")
print("=" * 60)

# Загружаем данные
df = pd.read_stata(DATA_PATH, convert_categoricals=False)
print(f"Загружено: {len(df)} строк, {len(df.columns)} столбцов")

# Заменяем пропуски
df.replace(MISSING_CODES, np.nan, inplace=True)

# Создаём возраст
df['age'] = 2024 - df['cch6']  # 2024 - год рождения

# Перекодируем пол
df['gender'] = df['cch5'].map({1: 'Мужской', 2: 'Женский'})

# Перекодируем образование
edu_map = {
    1: 'Начальное',
    2: 'Неполное среднее',
    3: 'Среднее',
    4: 'Среднее специальное',
    5: 'Высшее',
    6: 'Высшее и выше'
}
df['education'] = df['cc_diplom'].map(edu_map)

# Фильтруем по качеству (только хорошие интервью)
df = df[df['ccs3'] <= 1]  # хорошее понимание
df = df[df['ccs7'] != 3]  # не ненадёжные

print(f"После фильтрации: {len(df)} строк")

# Сохраняем
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(f"{OUTPUT_DIR}/rlms_clean.csv", index=False)
print(f"Сохранено: {OUTPUT_DIR}/rlms_clean.csv")
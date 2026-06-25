"""
create_profiles.py — создание демографических профилей
"""

import os
import json
import pandas as pd
import random
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Загружаем очищенные данные
df = pd.read_csv(f"{OUTPUT_DIR}/rlms_clean.csv")

def create_profiles(df, n=30):
    profiles = []
    
    # Выбираем случайных респондентов
    sampled = df.sample(min(n, len(df)))
    
    for idx, row in sampled.iterrows():
        profile = {
            "id": f"P_{len(profiles)+1:03d}",
            "demographics": {
                "gender": row.get("gender", "Не указано"),
                "age": int(row.get("age", 30)) if pd.notna(row.get("age")) else 30,
                "education": row.get("education", "Среднее"),
                "region": int(row.get("region", 1)) if pd.notna(row.get("region")) else 1,
                "has_children": row.get("ccj72.171", 2) == 1 if pd.notna(row.get("ccj72.171")) else False,
            },
            "socio_economic": {
                "working": row.get("ccj77", 2) == 1 if pd.notna(row.get("ccj77")) else False,
                "life_satisfaction": int(row.get("ccj65", 3)) if pd.notna(row.get("ccj65")) else 3,
            },
            "health": {
                "self_health": int(row.get("ccm3", 3)) if pd.notna(row.get("ccm3")) else 3,
            }
        }
        profiles.append(profile)
    
    return profiles

profiles = create_profiles(df, n=50)

# Сохраняем
with open(f"{OUTPUT_DIR}/profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2)

print(f"Создано {len(profiles)} профилей")
print(f"Сохранено: {OUTPUT_DIR}/profiles.json")
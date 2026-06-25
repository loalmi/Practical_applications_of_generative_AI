"""
generator.py — генератор 50 заявок на курсы ДПО.

Использует:
- make_client() с response_model=Application
- max_retries=3
- seed_city в промпте для борьбы с mode collapse
"""

import json
import csv
import random
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from llm_client import get_model, make_client
from schema import Application, CITIES

# Создаём клиент с поддержкой response_model
client = make_client()
MODEL = get_model()

N_APPLICATIONS = 50


def build_prompt(seed_city: str) -> tuple[str, str]:
    """
    Собирает промпты для генерации одной заявки.
    seed_city — случайный город, чтобы бороться с mode collapse.
    """
    system_prompt = f"""Ты — генератор синтетических заявок на курсы повышения квалификации (ДПО).
Создай реалистичную заявку от человека, который хочет пройти обучение.

ВАЖНО:
- Город: {seed_city} (обязательно укажи этот город)
- ФИО должно быть русским, реалистичным
- Возраст: от 22 до 65 лет
- Опыт работы: от 0 до 40 лет
- Год окончания вуза: от 1980 до 2024 (должен соответствовать возрасту)

Верни ответ ОДНИМ валидным JSON-объектом.
Никакого текста до или после JSON.
"""

    user_prompt = "Создай одну заявку на курс повышения квалификации."

    return system_prompt, user_prompt


def generate_one(seed_city: str) -> Application:
    """Генерирует одну заявку с фиксированным городом."""
    system_prompt, user_prompt = build_prompt(seed_city)

    application = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.9,
    )
    return application


def main():
    print(f"Генерация {N_APPLICATIONS} заявок...")
    print(f"Модель: {MODEL}")
    print(f"Городов в списке: {len(CITIES)}")
    print("-" * 50)

    applications = []
    failed = 0

    # Стратификация: равномерное распределение по городам
    # По 5 заявок на каждый из 10 городов (итого 50)
    cities_for_generation = CITIES[:10] * 5  # 10 городов × 5 раз = 50
    random.shuffle(cities_for_generation)  # Перемешиваем для разнообразия

    for i, city in enumerate(cities_for_generation, 1):
        print(f"[{i}/{N_APPLICATIONS}] Город: {city}...", end=" ")
        try:
            app = generate_one(city)
            applications.append(app)
            print(f"{app.full_name} ({app.speciality})")
        except Exception as e:
            failed += 1
            print(f"Ошибка: {type(e).__name__}: {str(e)[:50]}")
        time.sleep(0.3)

    print("-" * 50)
    print(f"Успешно: {len(applications)}")
    print(f"Ошибок: {failed}")

    # Сохраняем в CSV (address распакован в city и district)
    if applications:
        with open("applications.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "full_name",
                "age",
                "city",
                "district",
                "speciality",
                "desired_course",
                "years_of_experience",
                "graduation_year",
            ])

            for app in applications:
                writer.writerow([
                    app.full_name,
                    app.age,
                    app.address.city,
                    app.address.district,
                    app.speciality,
                    app.desired_course,
                    app.years_of_experience,
                    app.graduation_year,
                ])

        print(f"__Сохранено в applications.csv ({len(applications)} записей)")

        # Сохраняем сырые данные в JSON (для анализа)
        with open("applications.json", "w", encoding="utf-8") as f:
            json.dump(
                [app.model_dump() for app in applications],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("__Сохранено в applications.json")


if __name__ == "__main__":
    main()
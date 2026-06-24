"""
stress_test.py — стресс-тест на конфликт промпта и схемы.

Проверяем, что будет, если промпт говорит "придумай оригинальный курс",
а схема требует только строго определённые курсы из Literal.
"""

import time
from typing import Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from llm_client import get_model, make_client

client = make_client()
MODEL = get_model()

# ───── Конфликт: промпт просит "оригинальный курс", схема ограничивает ─────
CONFLICTING_PROMPT = """
Ты — генератор заявок на курсы повышения квалификации.
Создай реалистичную заявку от человека, который хочет пройти обучение.

ВАЖНО: Придумай ОРИГИНАЛЬНЫЙ, НЕСТАНДАРТНЫЙ курс, который не входит в обычные списки.
Например: "Квантовое программирование", "Нейроэкономика", "Космический туризм" и т.п.

Город: Москва.
ФИО должно быть русским.
Возраст: 30 лет.
Опыт работы: 5 лет.
Год окончания: 2015.
"""

# Схема разрешает ТОЛЬКО эти курсы
ALLOWED_COURSES = [
    "Python для анализа данных",
    "Машинное обучение",
    "Бизнес-аналитика",
    "Управление продуктом",
    "Цифровой маркетинг",
    "Кибербезопасность",
    "Data Science",
    "Инженер данных",
]

class ConflictApplication(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=22, le=65)
    city: str = Field(...)
    speciality: str = Field(...)
    desired_course: Literal[tuple(ALLOWED_COURSES)] = Field(...)  # 👈 ЖЁСТКОЕ ОГРАНИЧЕНИЕ
    years_of_experience: int = Field(..., ge=0, le=40)
    graduation_year: int = Field(..., ge=1980, le=2024)


def stress_test(max_retries: int):
    """Прогоняет стресс-тест с разными max_retries."""
    print(f"\n━━━ Стресс-тест: max_retries={max_retries} ━━━")
    print(f"Промпт: 'Придумай оригинальный курс'")
    print(f"Схема: desired_course ∈ {ALLOWED_COURSES[:3]}...")
    
    start_time = time.time()
    attempts = 0
    
    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CONFLICTING_PROMPT},
                {"role": "user", "content": "Создай одну заявку."},
            ],
            response_model=ConflictApplication,
            max_retries=max_retries,
            temperature=0.9,
        )
        elapsed = time.time() - start_time
        print(f"  УСПЕХ за {elapsed:.1f}с")
        print(f"  Курс: {result.desired_course}")
        print(f"  Имя: {result.full_name}")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ОШИБКА за {elapsed:.1f}с")
        print(f"  Потрачено запросов: ~{max_retries + 1}")
        print(f"  {type(e).__name__}: {str(e)[:200]}")
        return False, elapsed


def main():
    print(f"Модель: {MODEL}")
    print("=" * 60)
    print("СТРЕСС-ТЕСТ: конфликт промпта и схемы")
    print("Промпт: 'Придумай оригинальный курс'")
    print("Схема: desired_course ∈ строго определённый список")
    print("=" * 60)
    
    results = []
    
    # Проверяем разные max_retries
    for retries in [0, 1, 2, 3, 5]:
        success, elapsed = stress_test(retries)
        results.append({
            'max_retries': retries,
            'success': success,
            'elapsed': elapsed
        })
        time.sleep(0.5)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("-" * 60)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  max_retries={r['max_retries']}: {status} за {r['elapsed']:.1f}с")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("-" * 60)
    
    # Находим, при каком max_retries сдались
    failures = [r for r in results if not r['success']]
    if failures:
        last_fail = failures[-1]
        print(f"  ❌ При max_retries={last_fail['max_retries']} тест упал")
    
    successes = [r for r in results if r['success']]
    if successes:
        first_success = successes[0]
        print(f"  ✅ При max_retries={first_success['max_retries']} тест прошёл")
    
    print("\n Мораль:")
    print("     - Если промпт противоречит схеме, retry помогает НЕ ВСЕГДА")
    print("     - Увеличение max_retries не решает проблему, а только тратит токены")
    print("     - Правильное решение: согласовать промпт и схему")
    print("     - В нашем случае модель в итоге подчинилась схеме (Literal)")


if __name__ == "__main__":
    main()
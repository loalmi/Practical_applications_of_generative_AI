"""
schema.py — Pydantic-модели для заявок на курсы ДПО.

Содержит:
- Address — вложенная модель адреса
- Application — основная модель заявки
- @field_validator для проверки согласованности возраста и года окончания
"""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator

# Список городов (минимум 10)
CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Челябинск",
    "Самара",
    "Омск",
    "Ростов-на-Дону",
    "Уфа",
    "Красноярск",
]

# Специальности (минимум 8)
SPECIALITIES = [
    "Программист",
    "Аналитик данных",
    "Менеджер проектов",
    "Маркетолог",
    "Финансист",
    "HR-специалист",
    "Юрист",
    "Преподаватель",
    "Врач",
    "Архитектор",
]

# Желаемые курсы (минимум 6)
COURSES = [
    "Python для анализа данных",
    "Машинное обучение",
    "Бизнес-аналитика",
    "Управление продуктом",
    "Цифровой маркетинг",
    "Кибербезопасность",
    "Data Science",
    "Инженер данных",
]


class Address(BaseModel):
    """Вложенная модель адреса (раунд 4.5)"""

    city: Literal[tuple(CITIES)] = Field(..., description="Город проживания")
    district: str = Field(
        ..., min_length=2, max_length=50, description="Район или округ"
    )


class Application(BaseModel):
    """Модель заявки на курс повышения квалификации"""

    full_name: str = Field(..., min_length=2, max_length=100, description="ФИО полностью")
    age: int = Field(..., ge=22, le=65, description="Возраст")
    address: Address = Field(..., description="Адрес проживания")
    speciality: Literal[tuple(SPECIALITIES)] = Field(..., description="Текущая специальность")
    desired_course: Literal[tuple(COURSES)] = Field(..., description="Желаемый курс")
    years_of_experience: int = Field(..., ge=0, le=40, description="Опыт работы в годах")
    graduation_year: int = Field(..., ge=1980, le=2024, description="Год окончания вуза")

    @field_validator("graduation_year")
    @classmethod
    def validate_graduation_consistency(cls, v: int, info) -> int:
        """
        Проверяет, что год окончания вуза согласован с возрастом.
        Логика: человек оканчивает вуз примерно в 22-24 года.
        """
        age = info.data.get("age")
        if age is not None:
            current_year = datetime.now().year
            # Примерный год окончания = текущий год - возраст + 22 (средний возраст окончания)
            expected_graduation_lower = current_year - age + 19  # 19 лет — ранний выпуск
            expected_graduation_upper = current_year - age + 27  # 27 лет — поздний выпуск

            if not (expected_graduation_lower <= v <= expected_graduation_upper):
                raise ValueError(
                    f"Год окончания {v} не согласуется с возрастом {age}. "
                    f"Ожидаемый диапазон: {expected_graduation_lower}-{expected_graduation_upper}"
                )
        return v

    @field_validator("years_of_experience")
    @classmethod
    def validate_experience_consistency(cls, v: int, info) -> int:
        """
        Проверяет, что опыт работы не превышает возраст - 22.
        """
        age = info.data.get("age")
        if age is not None:
            max_experience = age - 22
            if v > max_experience:
                raise ValueError(
                    f"Опыт работы {v} лет не может превышать возраст {age} - 22 = {max_experience}"
                )
        return v
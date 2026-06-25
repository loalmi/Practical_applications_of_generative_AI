"""
survey.py — проведение опроса через LLM с валидацией
"""

import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from llm_client import ask_llm, get_model

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

# Загружаем профили
with open(f"{OUTPUT_DIR}/profiles.json", "r", encoding="utf-8") as f:
    profiles = json.load(f)

# Загружаем анкету
from questions import SURVEY_QUESTIONS


class SurveyAnswer(BaseModel):
    """Модель ответа с валидацией"""
    answer: str = Field(description="Выбранный вариант ответа")
    confidence: float = Field(default=0.8, ge=0, le=1, description="Уверенность в ответе")
    
    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Ответ не может быть пустым или слишком коротким")
        return v.strip()
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("Уверенность должна быть от 0 до 1")
        return v


def ask_question(profile: dict, question: dict) -> SurveyAnswer:
    """Задать один вопрос LLM с учётом профиля и вернуть валидированный ответ"""
    
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(question['options'])])
    
    prompt = f"""Ты — человек с таким демографическим профилем:
- Пол: {profile['demographics']['gender']}
- Возраст: {profile['demographics']['age']} лет
- Образование: {profile['demographics']['education']}
- Работает: {'Да' if profile['socio_economic']['working'] else 'Нет'}

Ответь на вопрос анкеты. Выбери ОДИН вариант из списка.

Вопрос: {question['question']}

Варианты:
{options_text}

Верни ответ строго в формате JSON:
{{"answer": "текст выбранного варианта", "confidence": 0.8}}"""

    resp = ask_llm(prompt, SurveyAnswer, temperature=0.3)
    return resp


def run_survey():
    """Прогнать анкету по всем профилям"""
    results = []
    total = len(profiles) * len(SURVEY_QUESTIONS)
    count = 0
    errors = 0

    print(f"Начинаем опрос {len(profiles)} респондентов...")
    print(f"Всего вопросов: {len(SURVEY_QUESTIONS)}")
    print(f"Всего ответов: {total}")
    print(f"Валидация включена (проверка ответов + confidence)")
    print()

    for p in profiles:
        profile_id = p["id"]
        profile_answers = {"profile_id": profile_id, "demographics": p["demographics"], "answers": []}

        for q in SURVEY_QUESTIONS:
            try:
                resp = ask_question(p, q)
                profile_answers["answers"].append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "answer": resp.answer,
                    "confidence": resp.confidence
                })
                count += 1
                print(f"  [{count}/{total}] {profile_id} -> {q['id']}: {resp.answer[:40]}... (conf={resp.confidence:.2f})")
            except Exception as e:
                errors += 1
                print(f"  Ошибка: {e}")
                profile_answers["answers"].append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "answer": f"(ошибка: {e})",
                    "confidence": 0.0
                })

            time.sleep(0.3)

        results.append(profile_answers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/survey_results_validated.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено: {OUTPUT_DIR}/survey_results_validated.json")
    print(f"Всего ответов: {count}")
    print(f"Ошибок: {errors}")
    print(f"Валидация: {'включена' if errors == 0 else f'было {errors} ошибок'}")


if __name__ == "__main__":
    run_survey()
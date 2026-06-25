"""
eval.py — оценка качества ответов LLM
"""

import json
import os
from dotenv import load_dotenv
from llm_client import ask_llm
from pydantic import BaseModel, Field

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")

with open(f"{OUTPUT_DIR}/survey_results_validated.json", "r", encoding="utf-8") as f:
    results = json.load(f)

from questions import SURVEY_QUESTIONS


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="Оценка от 1 до 5")
    comment: str = Field(..., description="Краткое объяснение")
    valid_option: bool = Field(..., description="Выбран ли допустимый вариант")
    matches_profile: bool = Field(..., description="Соответствует ли ответ профилю")


def evaluate_response(profile: dict, question: dict, answer: str) -> dict:
    """Оценивает один ответ с помощью LLM-as-judge"""
    
    prompt = f"""Ты — строгий судья. Оцени ответ респондента на вопрос анкеты.

Профиль респондента:
- Пол: {profile.get('gender', 'Не указано')}
- Возраст: {profile.get('age', 'Не указан')} лет
- Образование: {profile.get('education', 'Не указано')}

Вопрос: {question['question']}
Допустимые варианты: {', '.join(question['options'])}
Ответ респондента: {answer}

Оцени ответ по шкале 1-5:
1 — полностью не соответствует
2 — в основном не соответствует
3 — частично соответствует
4 — в основном соответствует
5 — полностью соответствует

Также укажи:
- Выбран ли допустимый вариант (true/false)
- Соответствует ли ответ профилю (true/false)

Верни JSON с полями: score, comment, valid_option, matches_profile"""

    return ask_llm(prompt, JudgeVerdict, temperature=0.3)


def run_eval():
    """Запускает оценку всех ответов"""
    eval_results = []
    total_tokens = 0
    total_calls = 0

    print("=" * 60)
    print("ЗАПУСК EVAL")
    print("=" * 60)

    respondents = results[:min(15, len(results))]
    
    for respondent in respondents:
        profile = respondent.get("demographics", {})
        
        for ans in respondent.get("answers", []):
            question = None
            for q in SURVEY_QUESTIONS:
                if q["id"] == ans["question_id"]:
                    question = q
                    break
            
            if not question:
                continue

            try:
                verdict = evaluate_response(profile, question, ans["answer"])
                total_calls += 1
            except Exception as e:
                print(f"  Ошибка: {e}")
                continue
            
            is_valid_option = ans["answer"] in question["options"]
            is_pass = verdict.valid_option and verdict.matches_profile and verdict.score >= 3

            tokens_used = len(ans["answer"].split()) + len(question["question"].split()) + 100
            total_tokens += tokens_used

            eval_results.append({
                "question": question["question"],
                "profile": profile,
                "model_answer": ans["answer"],
                "valid_option": is_valid_option,
                "judge_score": verdict.score,
                "judge_comment": verdict.comment,
                "matches_profile": verdict.matches_profile,
                "pass_fail": "PASS" if is_pass else "FAIL",
                "llm_calls": 1,
                "tokens_used": tokens_used,
                "estimated_cost": round(tokens_used / 1_000_000 * 0.014, 6)
            })

            print(f"  {question['id']}: {ans['answer'][:30]}... -> {'PASS' if is_pass else 'FAIL'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(eval_results),
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
            "estimated_cost": round(total_tokens / 1_000_000 * 0.014, 6),
            "results": eval_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nСохранено: {OUTPUT_DIR}/eval_results.json")
    print(f"Всего кейсов: {len(eval_results)}")
    print(f"Всего LLM-вызовов: {total_calls}")
    print(f"Токенов: {total_tokens}")
    print(f"Стоимость: ${round(total_tokens / 1_000_000 * 0.014, 6)}")


if __name__ == "__main__":
    run_eval()
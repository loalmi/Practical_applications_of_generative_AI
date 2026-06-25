"""
test_critic.py — Замер критики (ДЗ, часть 3)
Проверяет, как температура влияет на способность Критика находить ошибки.
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict

# 5 БИТЫХ НАБОРОВ ОТВЕТОВ

FAKE_BROKEN = [
    # 1. Арифметика без calculate
    {
        "name": "арифметика без calculate",
        "plan": Plan(
            reasoning="Нужно сравнить курсы USD и EUR",
            subquestions=[
                SubQuestion(id=1, question="Курс USD?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR?", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD?",
                answer="USD=82.5, EUR=89, разница=6.5",
                used_tools=["get_fx_rate"]
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс EUR?",
                answer="EUR=89",
                used_tools=["get_fx_rate"]
            ),
        }
    },
    # 2. Выдуманное число (галлюцинация)
    {
        "name": "выдуманное число",
        "plan": Plan(
            reasoning="Нужно узнать курс USD",
            subquestions=[
                SubQuestion(id=1, question="Курс USD?", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD?",
                answer="Курс USD = 85.0 (по данным ЦБ)",
                used_tools=["get_fx_rate"]
            ),
        }
    },
    # 3. Несогласованные данные
    {
        "name": "несогласованные данные",
        "plan": Plan(
            reasoning="Сравнить курс USD в 2022 и 2024",
            subquestions=[
                SubQuestion(id=1, question="Курс USD в 2022?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD в 2024?", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD в 2022?",
                answer="USD=74.29",
                used_tools=["get_fx_rate"]
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс USD в 2024?",
                answer="USD=92.55, вырос на 18.26 (разница 18.26)",
                used_tools=["get_fx_rate"]
            ),
        }
    },
    # 4. Неверный инструмент
    {
        "name": "неверный инструмент",
        "plan": Plan(
            reasoning="Узнать инфляцию",
            subquestions=[
                SubQuestion(id=1, question="Инфляция в 2023?", expected_tools=["get_inflation"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Инфляция в 2023?",
                answer="Инфляция = 7.5%",
                used_tools=["get_fx_rate"]
            ),
        }
    },
    # 5. Пустой ответ (ошибка)
    {
        "name": "пустой ответ",
        "plan": Plan(
            reasoning="Узнать курс USD",
            subquestions=[
                SubQuestion(id=1, question="Курс USD?", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD?",
                answer="(ошибка: agent exceeded max_iter)",
                used_tools=["get_fx_rate", "get_fx_rate", "get_fx_rate"]
            ),
        }
    },
]


def run_critic_test(temperature: float, runs: int = 10) -> dict:
    """Прогнать все битые кейсы с заданной температурой."""
    results = {}
    for case in FAKE_BROKEN:
        false_accepts = 0
        for _ in range(runs):
            verdict = critic_with_temp(
                question=case["plan"].reasoning,
                plan=case["plan"],
                answers=case["answers"],
                temperature=temperature
            )
            if verdict.ok:
                false_accepts += 1
        results[case["name"]] = false_accepts
    return results


def critic_with_temp(question: str, plan: Plan, answers: dict[int, WorkerAnswer], temperature: float) -> Verdict:
    """Обёртка для critic с заданной температурой."""
    from llm_client import make_client, get_model
    client = make_client()
    model = get_model()
    
    plan_lines = []
    for sq in plan.subquestions:
        tools = ",".join(sq.expected_tools) or "—"
        deps = f" depends_on={sq.depends_on}" if sq.depends_on else ""
        plan_lines.append(f"  {sq.id}. [{tools}]{deps}  «{sq.question}»")
    plan_text = "\n".join(plan_lines) or "  (пустой план)"

    ans_lines = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        tools = ",".join(a.used_tools) or "—"
        ans_lines.append(f"  {sq_id}. [{tools}] {a.answer}")
    answers_text = "\n".join(ans_lines) or "(ответов нет)"

    prompt = f"""Ты — критик мульти-агентной системы. Проверь ответы.

Исходный вопрос: {question}

План:
{plan_text}

Ответы Исполнителей:
{answers_text}

Проверь:
1. Все ли числа получены через calculate?
2. Согласованы ли числа между подвопросами?
3. Нет ли ошибок в ответах?

Вердикт: ok=True если всё чисто, иначе ok=False."""

    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_model=Verdict,
        temperature=temperature,
        max_retries=2,
    )


def main():
    print("=" * 60)
    print("ЗАМЕР КРИТИКИ: temperature 0.0 vs 0.7")
    print("=" * 60)
    
    print("\n[1] Прогон с temperature=0.0 (10 раз каждый кейс)...")
    results_0 = run_critic_test(temperature=0.0, runs=10)
    
    print("\n[2] Прогон с temperature=0.7 (10 раз каждый кейс)...")
    results_7 = run_critic_test(temperature=0.7, runs=10)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ (ложные принятия / 10)")
    print("=" * 60)
    print(f"{'Битый кейс':<35} | T=0.0 | T=0.7")
    print("-" * 60)
    for case in FAKE_BROKEN:
        n = case["name"]
        print(f"{n:<35} | {results_0.get(n, 0):>5} | {results_7.get(n, 0):>5}")
    
    total_0 = sum(results_0.values())
    total_7 = sum(results_7.values())
    print("-" * 60)
    print(f"{'ИТОГО':<35} | {total_0:>5} | {total_7:>5}")
    print("=" * 60)
    
    print("\nВЫВОД:")
    if total_7 < total_0:
        print("  Гипотеза подтверждается: при T=0.7 Критик реже ошибается")
        print(f"     (T=0.0: {total_0}/50, T=0.7: {total_7}/50)")
    else:
        print("  Гипотеза не подтвердилась: T=0.7 не снизила количество ошибок")


if __name__ == "__main__":
    main()
"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, validate_plan
from worker import worker


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Вернуть список уровней (каждый уровень — список подвопросов)."""
    by_id = {s.id: s for s in subqs}
    in_degree = {s.id: 0 for s in subqs}
    for sq in subqs:
        for dep in sq.depends_on:
            if dep in by_id:
                in_degree[sq.id] += 1

    levels = []
    current = [sq for sq in subqs if in_degree[sq.id] == 0]
    while current:
        levels.append(current)
        next_level = []
        for sq in current:
            for other in subqs:
                if sq.id in other.depends_on:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        next_level.append(other)
        current = next_level
    return levels


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше."""
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        if sq.id not in visited:
            visit(sq.id, [])
    return ordered





def execute_level(level: list[SubQuestion], prev_answers: dict) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня параллельно."""
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, sq, prev_answers) for sq in level]
        results = [f.result() for f in futures]
    return {r.subquestion_id: r for r in results}


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [answers[i].answer for i in sorted(answers)]
    return " · ".join(parts)

import time
def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    validate: bool = False,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    start_time = time.time()
    
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    
    # Валидация плана (ДЗ, часть 1)
    if validate:
        errors = validate_plan(plan)
        if errors:
            if verbose:
                print(f"  [validator] Ошибки в плане: {errors}")
            plan = planner(question, feedback=f"План содержит несуществующие инструменты: {errors}")
            errors2 = validate_plan(plan)
            if errors2:
                if verbose:
                    print(f"  [validator] Повторная ошибка: {errors2}")
                plan.subquestions = []
                plan.reasoning = f"Не удалось построить корректный план. Ошибки: {errors2}"

    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}

        levels = _topological_levels(plan.subquestions)

        for level in levels:
            level_answers = execute_level(level, answers)
            answers.update(level_answers)
            for sq in level:
                ans = level_answers.get(sq.id)
                if ans is None:
                    continue
                trace.append({
                    "iter": iter_num,
                    "kind": "worker",
                    "sq_id": sq.id,
                    "used_tools": ans.used_tools,
                    "answer": ans.answer,
                })
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        # ordered = _topological_sort(plan.subquestions)
        # for sq in ordered:
        #     ans = worker(sq, prev_answers=answers)
        #     answers[sq.id] = ans
        #     trace.append(
        #         {
        #             "iter": iter_num,
        #             "kind": "worker",
        #             "sq_id": sq.id,
        #             "used_tools": ans.used_tools,
        #             "answer": ans.answer,
        #         }
        #     )
        #     if verbose:
        #         print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            elapsed = time.time() - start_time
            print(f"\n__Время выполнения: {elapsed:.2f} секунд")
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        # TODO: replan/rework
        break

    elapsed = time.time() - start_time
    print(f"\n__Время выполнения (с ошибкой): {elapsed:.2f} секунд")

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--validate", action="store_true", help="Включить валидатор")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet, validate=args.validate,)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
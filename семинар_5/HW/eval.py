"""
Мини-оценка: 4 вопроса, проверяем:
1. Что агент завершает работу за разумное число шагов.
2. Что в трассе шагов есть ожидаемые инструменты.
3. Что в финальном ответе упомянуты ожидаемые ключевые числа (опционально).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],  # число не фиксируем — зависит от живого запроса
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика. Классический многостадийный кейс.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы.",
    },
    
    # === НОВЫЕ 6 ВОПРОСОВ ===
    
    {
        "id": 5,
        "query": "Во сколько раз вырос курс евро с января 2020 по январь 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": ["ratio", "евро", "2020-01", "2026-01"],
        "comment": "Требует compare_periods. Сравнение курса евро за 6 лет.",
    },
    {
        "id": 6,
        "query": "Как изменилась инфляция в России за последние 5 лет? Сравни декабрь 2020 и декабрь 2025.",
        "expected_tools": ["compare_periods"],
        "must_have": ["delta", "инфляция", "2020-12", "2025-12"],
        "comment": "Требует compare_periods. Сравнение инфляции с разницей в 5 лет.",
    },
    {
        "id": 7,
        "query": "Какая была безработица в марте 2023?",
        "expected_tools": ["get_unemployment"],
        "must_have": ["%", "2023-03"],
        "comment": "ТРУДНЫЙ: дата может отсутствовать в данных (если данные только ежемесячные, а не ежедневные). Агент может ошибиться с форматом даты.",
    },
    {
        "id": 8,
        "query": "Сколько стоил юань в феврале 2022 и сколько он стоит сейчас? Насколько он вырос?",
        "expected_tools": ["get_fx_rate", "get_fx_rate", "calculate"],
        "must_have": ["юань", "CNY", "вырос"],
        "comment": "ТРУДНЫЙ: нужно распарсить 'сейчас' (сегодняшняя дата) и 'февраль 2022' (неоднозначный формат). Может ошибиться с определением 'сейчас'.",
    },
    {
        "id": 9,
        "query": "Какая реальная доходность по вкладу при текущей ключевой ставке, если инфляция за прошлый год составила 8%?",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["реальная", "доходность", "%"],
        "comment": "РЕАЛЬНЫЙ: классический макро-вопрос для инвестора. Реальная ставка = номинальная - инфляция.",
    },
    {
        "id": 10,
        "query": "Сравни курс доллара и евро сегодня с курсами годичной давности. Какая валюта укрепилась сильнее?",
        "expected_tools": ["get_fx_rate", "get_fx_rate", "get_fx_rate", "get_fx_rate", "compare"],
        "must_have": ["доллар", "евро", "укрепилась", "сильнее"],
        "comment": "РЕАЛЬНЫЙ: сложный сравнение двух валют за два периода. Требует 4 вызова get_fx_rate + логический вывод.",
    },
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(f"\n  tools used : {used_tools}")
    print(
        f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}"
    )
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  → {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Мини-оценка макро-агента")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы — видно повторные вызовы",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам",
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено")
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(f"  {mark} Q{r['id']} ({r['steps']} шагов) — {r['query'][:60]}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(
            f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений "
            f"к инструментам — столько вызовов ЦБ/Росстата сэкономлено."
        )

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()

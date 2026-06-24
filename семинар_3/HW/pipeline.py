"""
pipeline.py — полный конвейер для анализа лекций
Адаптировано под транскрипты лекций курса "Практическое применение генеративного ИИ"

Запуск:
    python pipeline.py
    python pipeline.py input/lecture_01.txt output/lecture_01
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from llm_client import get_model, make_client
from schema import (
    Expert,
    ExpertAspectAnalysis,
    AspectSentiment,
    LectureSummary,
    HierarchicalSummary,
    GroupSummary,
    JudgeReport,
    Verdict,
    MultiDocSummary,
)
from prompts import (
    IE_SYSTEM,
    ASPECTS_SYSTEM,
    DISCOVER_SYSTEM,
    CHUNK_SYSTEM,
    REDUCE_SYSTEM,
    GROUP_REDUCE_SYSTEM,
    JUDGE_SYSTEM,
    MULTI_DOC_SYSTEM,
)

client = make_client()
MODEL = get_model()


# ══════════════════════════════════════════════════════════
# Подсчёт токенов и стоимости
# ══════════════════════════════════════════════════════════

def estimate_cost(usage) -> dict:
    """Подсчёт стоимости на основе usage из ответа API.
    
    Цены DeepSeek-V4-flash (за 1M токенов):
    - prompt: $0.014
    - completion: $0.028
    """
    prompt_price_per_m = 0.014
    completion_price_per_m = 0.028
    
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    prompt_cost = (prompt_tokens / 1_000_000) * prompt_price_per_m
    completion_cost = (completion_tokens / 1_000_000) * completion_price_per_m
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(completion_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction (эксперт → тезисы)
# ══════════════════════════════════════════════════════════

def extract_expert(transcript: str, date: Optional[str] = None) -> Expert:
    """Извлечение эксперта и его тезисов из лекции"""
    return client.chat.completions.create(
        model=MODEL,
        response_model=Expert,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": transcript},
        ],
    )


def extract_expert_with_usage(transcript: str, date: Optional[str] = None) -> tuple[Expert, dict]:
    """Извлечение эксперта с возвратом usage"""
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=Expert,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    return result, estimate_cost(completion.usage)


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════

def extract_aspects(transcript: str, expert_name: str, date: str) -> ExpertAspectAnalysis:
    """Аспектный анализ лекции"""
    return client.chat.completions.create(
        model=MODEL,
        response_model=ExpertAspectAnalysis,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": transcript},
        ],
    )


def extract_aspects_with_usage(transcript: str, expert_name: str, date: str) -> tuple[ExpertAspectAnalysis, dict]:
    """Аспектный анализ с возвратом usage"""
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=ExpertAspectAnalysis,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        with_completion=True,
    )
    return result, estimate_cost(completion.usage)


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery
# ══════════════════════════════════════════════════════════

def discover_aspects(transcript: str) -> list[str]:
    """Автоматическое обнаружение тем в лекции"""
    response = client.chat.completions.create(
        model=MODEL,
        response_model=dict,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": DISCOVER_SYSTEM},
            {"role": "user", "content": transcript},
        ],
    )
    if "aspects" in response:
        return [a["name"] for a in response["aspects"]]
    return []


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce
# ══════════════════════════════════════════════════════════

def chunk_summary(fragment: str) -> dict:
    """Резюме по фрагменту лекции (MAP)"""
    return client.chat.completions.create(
        model=MODEL,
        response_model=dict,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": fragment},
        ],
    )


def reduce_summary(chunks: list[dict]) -> LectureSummary:
    """Финальная сводка по лекции (REDUCE)"""
    chunks_text = "\n\n".join(
        f"Фрагмент {i+1}: {json.dumps(c, ensure_ascii=False, indent=2)}"
        for i, c in enumerate(chunks)
    )
    return client.chat.completions.create(
        model=MODEL,
        response_model=LectureSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": chunks_text},
        ],
    )


def reduce_summary_with_usage(chunks: list[dict]) -> tuple[LectureSummary, dict]:
    """Финальная сводка с возвратом usage"""
    chunks_text = "\n\n".join(
        f"Фрагмент {i+1}: {json.dumps(c, ensure_ascii=False, indent=2)}"
        for i, c in enumerate(chunks)
    )
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=LectureSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": chunks_text},
        ],
        with_completion=True,
    )
    return result, estimate_cost(completion.usage)


# ══════════════════════════════════════════════════════════
# Раунд 3.5 — Иерархический Map-Reduce
# ══════════════════════════════════════════════════════════

def hierarchical_summary_full(transcript: str, chunk_size: int = 2000) -> HierarchicalSummary:
    """Трёхуровневый иерархический Map-Reduce."""
    from collections import defaultdict
    
    print("   [HMR] Уровень 1: MAP...")
    chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:
            continue
        try:
            summary = chunk_summary(chunk)
            key_points = summary.get("key_points", [])
            if not key_points:
                key_points = [f"Тезис из фрагмента {i+1}"]
            chunk_summaries.append({
                "index": i,
                "topic": summary.get("topic", f"Фрагмент {i+1}"),
                "key_points": key_points,
                "has_examples": summary.get("has_examples", False),
            })
        except Exception as e:
            print(f"      [HMR] Ошибка в чанке {i}: {e}")
            chunk_summaries.append({
                "index": i,
                "topic": f"Фрагмент {i+1}",
                "key_points": [f"Тезис из фрагмента {i+1}"],
                "has_examples": False,
            })
    
    if not chunk_summaries:
        return HierarchicalSummary(
            headline="Недостаточно данных для иерархической сводки",
            key_findings=["Текст слишком короткий или не удалось извлечь тезисы"],
            group_summaries=[],
            practical_applications=[],
            recommended_resources=[],
            overall_quality="low",
        )
    
    print(f"      Обработано чанков: {len(chunk_summaries)}")
    
    print("   [HMR] Уровень 2: GROUP...")
    groups = defaultdict(list)
    for cs in chunk_summaries:
        topic_words = cs["topic"].lower().split()[:3]
        key = " ".join(topic_words) if topic_words else "other"
        groups[key].append(cs)
    
    group_summaries = []
    for group_key, items in groups.items():
        all_points = []
        for item in items:
            all_points.extend(item["key_points"])
        unique_points = list(dict.fromkeys(all_points))[:5]
        if not unique_points:
            unique_points = [f"Тезис по теме {group_key}"]
        group_summaries.append({
            "theme": group_key,
            "fragments": [f"Фрагмент {c['index']+1}" for c in items],
            "key_points": unique_points,
            "overall_sentiment": "neutral",
        })
    
    print(f"      Сформировано групп: {len(group_summaries)}")
    
    print("   [HMR] Уровень 3: REDUCE...")
    all_points = []
    for g in group_summaries:
        all_points.extend(g["key_points"])
    unique_all_points = list(dict.fromkeys(all_points))[:8]
    if not unique_all_points:
        unique_all_points = ["Не удалось извлечь ключевые выводы"]
    
    headline = f"Лекция: {len(chunks)} фрагментов, {len(group_summaries)} тем"
    key_findings = unique_all_points[:6]
    
    return HierarchicalSummary(
        headline=headline,
        key_findings=key_findings,
        group_summaries=[GroupSummary(**g) for g in group_summaries],
        practical_applications=[],
        recommended_resources=[],
        overall_quality="high" if len(chunks) > 3 else "medium",
    )


def hierarchical_summary_full_with_usage(transcript: str, chunk_size: int = 3000) -> tuple[HierarchicalSummary, dict]:
    """Hierarchical MR с возвратом usage (суммарный по всем вызовам chunk_summary)"""
    from collections import defaultdict
    
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost_usd": 0.0,
        "completion_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }
    
    print("   [HMR] Уровень 1: MAP...")
    chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 100:
            continue
        try:
            summary, usage = chunk_summary_with_usage(chunk)
            for key in total_usage:
                if key in usage:
                    total_usage[key] += usage[key]
            key_points = summary.get("key_points", [])
            if not key_points:
                key_points = [f"Тезис из фрагмента {i+1}"]
            chunk_summaries.append({
                "index": i,
                "topic": summary.get("topic", f"Фрагмент {i+1}"),
                "key_points": key_points,
                "has_examples": summary.get("has_examples", False),
            })
        except Exception as e:
            print(f"      [HMR] Ошибка в чанке {i}: {e}")
            chunk_summaries.append({
                "index": i,
                "topic": f"Фрагмент {i+1}",
                "key_points": [f"Тезис из фрагмента {i+1}"],
                "has_examples": False,
            })
    
    if not chunk_summaries:
        return HierarchicalSummary(
            headline="Недостаточно данных для иерархической сводки",
            key_findings=["Текст слишком короткий или не удалось извлечь тезисы"],
            group_summaries=[],
            practical_applications=[],
            recommended_resources=[],
            overall_quality="low",
        ), total_usage
    
    print(f"      Обработано чанков: {len(chunk_summaries)}")
    
    print("   [HMR] Уровень 2: GROUP...")
    groups = defaultdict(list)
    for cs in chunk_summaries:
        topic_words = cs["topic"].lower().split()[:3]
        key = " ".join(topic_words) if topic_words else "other"
        groups[key].append(cs)
    
    group_summaries = []
    for group_key, items in groups.items():
        all_points = []
        for item in items:
            all_points.extend(item["key_points"])
        unique_points = list(dict.fromkeys(all_points))[:5]
        if not unique_points:
            unique_points = [f"Тезис по теме {group_key}"]
        group_summaries.append({
            "theme": group_key,
            "fragments": [f"Фрагмент {c['index']+1}" for c in items],
            "key_points": unique_points,
            "overall_sentiment": "neutral",
        })
    
    print(f"      Сформировано групп: {len(group_summaries)}")
    
    print("   [HMR] Уровень 3: REDUCE...")
    all_points = []
    for g in group_summaries:
        all_points.extend(g["key_points"])
    unique_all_points = list(dict.fromkeys(all_points))[:8]
    if not unique_all_points:
        unique_all_points = ["Не удалось извлечь ключевые выводы"]
    
    headline = f"Лекция: {len(chunks)} фрагментов, {len(group_summaries)} тем"
    key_findings = unique_all_points[:6]
    
    return HierarchicalSummary(
        headline=headline,
        key_findings=key_findings,
        group_summaries=[GroupSummary(**g) for g in group_summaries],
        practical_applications=[],
        recommended_resources=[],
        overall_quality="high" if len(chunks) > 3 else "medium",
    ), total_usage


def chunk_summary_with_usage(fragment: str) -> tuple[dict, dict]:
    """Резюме по фрагменту с возвратом usage"""
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=dict,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": fragment},
        ],
        with_completion=True,
    )
    return result, estimate_cost(completion.usage)


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════

def judge(claims: list[dict], summary: dict) -> JudgeReport:
    """Оценка качества лекции судьёй"""
    claims_text = json.dumps(claims, ensure_ascii=False, indent=2)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    return client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Тезисы:\n{claims_text}\n\nСводка:\n{summary_text}",
            },
        ],
    )


def judge_with_usage(claims: list[dict], summary: dict) -> tuple[JudgeReport, dict]:
    """Оценка качества с возвратом usage"""
    claims_text = json.dumps(claims, ensure_ascii=False, indent=2)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Тезисы:\n{claims_text}\n\nСводка:\n{summary_text}",
            },
        ],
        with_completion=True,
    )
    return result, estimate_cost(completion.usage)


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка
# ══════════════════════════════════════════════════════════

def multi_doc_summary(lectures: list[dict]) -> MultiDocSummary:
    """Сводка по нескольким лекциям"""
    lectures_text = "\n\n".join(
        f"## {l.get('title', f'Лекция {i+1}')}\n"
        + json.dumps(l, ensure_ascii=False, indent=2)
        for i, l in enumerate(lectures)
    )
    return client.chat.completions.create(
        model=MODEL,
        response_model=MultiDocSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": MULTI_DOC_SYSTEM},
            {"role": "user", "content": lectures_text},
        ],
    )


# ══════════════════════════════════════════════════════════
# Проверка цитат (check_quotes)
# ══════════════════════════════════════════════════════════

def check_quotes(aspects: list[AspectSentiment], transcript: str) -> list[tuple[str, str]]:
    """Проверяет, есть ли цитаты из аспектов в транскрипте."""
    t = transcript.lower()
    ghosts = []
    for a in aspects:
        probe = a.quote.strip().lower()[:30]
        if probe and probe not in t:
            ghosts.append((a.aspect, a.quote))
    return ghosts


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка (обёртка)
# ══════════════════════════════════════════════════════════

def run_multi_doc(output_dir: str = "output") -> None:
    """Собирает все сводки из output/ и формирует MultiDocSummary"""
    out_path = Path(output_dir)
    summaries = []
    
    for lecture_dir in out_path.iterdir():
        if lecture_dir.is_dir():
            summary_file = lecture_dir / f"{lecture_dir.name}_summary.json"
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    summaries.append({
                        "title": lecture_dir.name,
                        "headline": summary.get("headline", ""),
                        "key_findings": summary.get("key_findings", []),
                    })
    
    if len(summaries) < 2:
        print("Недостаточно сводок для multi-doc (нужно минимум 2)")
        return
    
    result = multi_doc_summary(summaries)
    print("\nMulti-doc сводка по всем лекциям:")
    print(f"   Общий заголовок: {result.overall_headline}")
    print("\n   Общие темы:")
    for theme in result.common_themes:
        print(f"      {theme}")
    print("\n   Уникальное по лекциям:")
    for title, unique in result.unique_per_source.items():
        print(f"      [{title}] {', '.join(unique)}")
    
    with open(out_path / "multi_doc_summary.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {out_path}/multi_doc_summary.json")


# ══════════════════════════════════════════════════════════
# Полный пайплайн
# ══════════════════════════════════════════════════════════

def analyze(transcript_path: str, out_dir: str = "output") -> None:
    """Полный конвейер анализа одной лекции"""
    start_time = time.time()
    
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    transcript = Path(transcript_path).read_text(encoding="utf-8")
    file_name = Path(transcript_path).stem

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost_usd": 0.0,
        "completion_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }

    print(f"Анализ: {file_name}")
    print("-" * 50)

    # 1. Извлечение эксперта
    print("Раунд 1: извлечение эксперта...")
    expert, usage = extract_expert_with_usage(transcript)
    for key in total_usage:
        if key in usage:
            total_usage[key] += usage[key]
    expert_data = expert.model_dump()
    (out / f"{file_name}_expert.json").write_text(
        json.dumps(expert_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"   Эксперт: {expert.name}, тезисов: {len(expert.claims)}")

    # 2. Аспектный анализ
    print("Раунд 2: аспектный анализ...")
    aspects, usage = extract_aspects_with_usage(transcript, expert.name, expert.date)
    for key in total_usage:
        if key in usage:
            total_usage[key] += usage[key]
    aspects_data = aspects.model_dump()
    (out / f"{file_name}_aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"   Аспектов: {len(aspects.aspects)}")

    # Проверка цитат
    print("Проверка цитат...")
    ghosts = check_quotes(aspects.aspects, transcript)
    if ghosts:
        print(f"   Найдено {len(ghosts)} ghost-цитат:")
        for aspect, quote in ghosts[:3]:
            print(f"      [{aspect}] {quote[:60]}...")
    else:
        print("   Все цитаты подтверждены")

    # 3. Сводка (упрощённый MR)
    print("Раунд 3: сводка...")
    chunks = [{"fragment": transcript[:3000]}, {"fragment": transcript[3000:6000]}]
    summary, usage = reduce_summary_with_usage(chunks)
    for key in total_usage:
        if key in usage:
            total_usage[key] += usage[key]
    summary_data = summary.model_dump()
    (out / f"{file_name}_summary.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"   Сводка: {summary.headline}")

    # 3.5. Hierarchical MR (Раунд 3.5) — только для длинных текстов
    print("→ Раунд 3.5: иерархический Map-Reduce...")
    if len(transcript) > 10000:
        hmr_result, usage = hierarchical_summary_full_with_usage(transcript, chunk_size=3000)
        for key in total_usage:
            if key in usage:
                total_usage[key] += usage[key]
        hmr_data = hmr_result.model_dump()
        (out / f"{file_name}_hierarchical_mr.json").write_text(
            json.dumps(hmr_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"   Чанков: {len(hmr_result.group_summaries)}, групп: {len(hmr_result.group_summaries)}")
        print(f"   Ключевых выводов: {len(hmr_result.key_findings)}")
    else:
        print("   ⏭Пропущено (текст слишком короткий)")

    # 4. Судья
    print("Раунд 5: судья...")
    claims_data = [c.model_dump() for c in expert.claims]
    report, usage = judge_with_usage(claims_data, summary_data)
    for key in total_usage:
        if key in usage:
            total_usage[key] += usage[key]
    report_data = report.model_dump()
    (out / f"{file_name}_judge.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"   Оценка судьи: {report.overall_score:.2f}")

    # 5. CSV для аспектов
    aspects_df = pd.DataFrame([
        {
            "expert": aspects.expert_name,
            "date": aspects.date,
            "aspect": a.aspect,
            "sentiment": a.sentiment,
            "quote": a.quote,
            "confidence": a.confidence,
        }
        for a in aspects.aspects
    ])
    aspects_df.to_csv(out / f"{file_name}_aspects.csv", index=False, encoding="utf-8")

    # Сохраняем статистику использования
    elapsed_time = time.time() - start_time
    total_usage["elapsed_time_seconds"] = round(elapsed_time, 1)
    
    (out / f"{file_name}_usage.json").write_text(
        json.dumps(total_usage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print(f"\n📊 Статистика использования:")
    print(f"   Время: {total_usage['elapsed_time_seconds']:.1f}с")
    print(f"   Токенов: {total_usage['total_tokens']:,}")
    print(f"   Стоимость: ${total_usage['total_cost_usd']:.6f}")

    print("\n" + "=" * 50)
    print(f"Готово! Артефакты в: {out}/")
    print(f"   {file_name}_expert.json")
    print(f"   {file_name}_aspects.json")
    print(f"   {file_name}_summary.json")
    print(f"   {file_name}_hierarchical_mr.json")
    print(f"   {file_name}_judge.json")
    print(f"   {file_name}_aspects.csv")
    print(f"   {file_name}_usage.json")


# ══════════════════════════════════════════════════════════
# Запуск
# ══════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        input_dir = Path("input")
        if not input_dir.exists():
            print("Папка input/ не найдена. Создайте её и положите туда лекции.")
            print("Использование: python pipeline.py <путь_к_лекции.txt> [папка_выхода]")
            sys.exit(1)

        lectures = list(input_dir.glob("*.txt"))
        if not lectures:
            print("В папке input/ нет .txt файлов с лекциями.")
            sys.exit(1)

        print(f"Найдено лекций: {len(lectures)}")
        for i, lecture in enumerate(lectures):
            print(f"\n--- Лекция {i+1}: {lecture.name} ---")
            analyze(str(lecture), f"output/{lecture.stem}")

        print("\n" + "=" * 50)
        print("Создание multi-doc сводки...")
        run_multi_doc("output")

        print("\n" + "=" * 50)
        print("Все лекции обработаны!")
    else:
        transcript_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
        analyze(transcript_path, out_dir)


if __name__ == "__main__":
    main()
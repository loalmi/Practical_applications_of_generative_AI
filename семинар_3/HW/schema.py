"""
schema.py — общие Pydantic-схемы пайплайна
===========================================
Адаптировано под транскрипты первых пяти лекций курса "Практическое применение генеративного ИИ"
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════
class Claim(BaseModel):
    """Тезис/утверждение эксперта"""
    topic: str = Field(..., min_length=3, description="Тема тезиса")
    content: str = Field(..., min_length=5, description="Содержание тезиса")
    example: Optional[str] = Field(default=None, description="Пример из лекции")
    importance: Literal["high", "medium", "low"] = Field(default="medium")


class Expert(BaseModel):
    """Модель эксперта (лектора)"""
    name: str = Field(..., description="Имя эксперта")
    date: str = Field(..., description="Дата лекции")
    claims: list[Claim] = Field(default_factory=list, description="Ключевые тезисы")
    references: list[str] = Field(default_factory=list, description="Упоминания источников")

# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════
class AspectSentiment(BaseModel):
    """Оценка по аспекту"""
    aspect: Literal[
        "новизна",          # насколько материал новый, актуальный
        "обоснованность",   # есть ли аргументы, доказательства
        "практичность",     # можно ли применить на практике
        "риски",            # какие риски описаны или подразумеваются
        "ясность"           # насколько понятно изложен материал
    ]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(..., min_length=3, description="Цитата из лекции")
    confidence: float = Field(ge=0, le=1)


class ExpertAspectAnalysis(BaseModel):
    """Аспектный анализ лекции"""
    expert_name: str = Field(..., description="Имя эксперта")
    date: str = Field(..., description="Дата лекции")
    aspects: list[AspectSentiment] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════
class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)

# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce-резюме
# ══════════════════════════════════════════════════════════
class ChunkSummary(BaseModel):
    """Сводка по фрагменту лекции"""
    topic: str = Field(..., description="Тема фрагмента")
    key_points: list[str] = Field(min_length=1, max_length=6)
    has_examples: bool = False


class LectureSummary(BaseModel):
    """Общая сводка по лекции"""
    headline: str = Field(..., description="Заголовок-резюме")
    key_findings: list[str] = Field(min_length=2, max_length=8)
    practical_applications: list[str] = Field(default_factory=list)
    recommended_resources: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════
# Раунд 3.5 — Иерархический Map-Reduce
# ══════════════════════════════════════════════════════════
class HierarchicalChunkSummary(BaseModel):
    """Сводка по фрагменту лекции (уровень 1)"""
    topic: str = Field(..., description="Тема фрагмента")
    key_points: list[str] = Field(min_length=1, max_length=5, description="Ключевые тезисы")
    examples: list[str] = Field(default_factory=list, description="Примеры из фрагмента")
    confidence: float = Field(ge=0, le=1, description="Уверенность в извлечении")


class GroupSummary(BaseModel):
    """Групповая сводка по нескольким фрагментам (уровень 2)"""
    theme: str = Field(..., description="Общая тема группы фрагментов")
    fragments: list[str] = Field(..., description="Названия фрагментов в группе")
    key_points: list[str] = Field(min_length=1, max_length=6, description="Общие ключевые тезисы")
    overall_sentiment: Literal["positive", "negative", "mixed", "neutral"] = Field(default="neutral")


class HierarchicalSummary(BaseModel):
    """Итоговая сводка по всей лекции (уровень 3)"""
    headline: str = Field(..., description="Заголовок-резюме всей лекции")
    key_findings: list[str] = Field(min_length=2, max_length=8, description="Ключевые выводы")
    group_summaries: list[GroupSummary] = Field(default_factory=list, description="Сводки по группам тем")
    practical_applications: list[str] = Field(default_factory=list, description="Практические применения")
    recommended_resources: list[str] = Field(default_factory=list, description="Рекомендуемые источники")
    overall_quality: Literal["high", "medium", "low"] = Field(default="medium", description="Общая оценка качества")


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════
class Verdict(BaseModel):
    """Вердикт по одному тезису"""
    thesis: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    """Отчёт судьи"""
    verdicts: list[Verdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str

# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка
# ══════════════════════════════════════════════════════════
class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_source: dict[str, list[str]]
    overall_headline: str
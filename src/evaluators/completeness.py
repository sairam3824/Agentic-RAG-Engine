from __future__ import annotations

from src.utils.text import keyword_set


def score_completeness(query: str, docs: list[dict], query_type: str) -> float:
    if not docs:
        return 0.0

    expected_docs = {
        "factual": 2,
        "analytical": 4,
        "comparison": 4,
        "how-to": 3,
    }.get(query_type, 3)

    query_keywords = keyword_set(query)
    if not query_keywords:
        return 0.6

    coverage: set[str] = set()
    for doc in docs[:6]:
        doc_keywords = keyword_set(doc.get("content", ""))
        coverage.update(query_keywords & doc_keywords)

    keyword_coverage = len(coverage) / len(query_keywords)
    depth_score = min(len(docs) / expected_docs, 1.0)
    return round((keyword_coverage * 0.65) + (depth_score * 0.35), 3)


def score_freshness(query: str, docs: list[dict], used_sources: list[str]) -> float:
    if not docs:
        return 0.0

    freshness_keywords = {"today", "latest", "current", "recent", "2026", "2025", "yesterday"}
    needs_fresh = any(token in query.lower() for token in freshness_keywords)

    has_web = any(not source.startswith("sqlite:") and source.startswith("http") for source in used_sources)
    if needs_fresh:
        return 0.95 if has_web else 0.35
    return 0.8 if docs else 0.0

from __future__ import annotations

from src.utils.text import overlap_ratio


def score_relevance(query: str, docs: list[dict]) -> float:
    if not docs:
        return 0.0
    top_docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)[:5]
    overlaps = [overlap_ratio(query, doc.get("content", "")) for doc in top_docs]
    model_scores = [float(doc.get("score", 0.0) or 0.0) for doc in top_docs]
    combined = [(o * 0.7) + (min(s, 1.0) * 0.3) for o, s in zip(overlaps, model_scores)]
    return round(sum(combined) / len(combined), 3)

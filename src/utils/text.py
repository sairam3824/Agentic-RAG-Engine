from __future__ import annotations

import re
from typing import Iterable

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def keyword_set(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1}


def overlap_ratio(left: str, right: str) -> float:
    left_tokens = keyword_set(left)
    if not left_tokens:
        return 0.0
    right_tokens = keyword_set(right)
    return len(left_tokens & right_tokens) / len(left_tokens)


def dedupe_by_key(items: Iterable[dict], key: str) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        value = str(item.get(key, ""))
        if value in seen:
            continue
        seen.add(value)
        unique.append(item)
    return unique

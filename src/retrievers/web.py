from __future__ import annotations

import os
from typing import Any


class TavilyWebRetriever:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None

        if self.api_key:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self._client:
            return []
        try:
            response = self._client.search(query=query, max_results=k, include_answer=False)
            results = response.get("results", [])
            docs: list[dict[str, Any]] = []
            for idx, item in enumerate(results):
                docs.append(
                    {
                        "id": f"web-{idx}",
                        "content": item.get("content", ""),
                        "source": item.get("url", "web"),
                        "score": float(item.get("score", 0.0) or 0.0),
                        "metadata": {
                            "title": item.get("title", ""),
                            "published_date": item.get("published_date", ""),
                        },
                    }
                )
            return docs
        except Exception:
            return []

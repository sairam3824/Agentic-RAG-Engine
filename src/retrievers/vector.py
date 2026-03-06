from __future__ import annotations

import os
import uuid
from typing import Any

from src.utils.text import dedupe_by_key, overlap_ratio


class VectorRetriever:
    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "agentic-rag-engine") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._cache: list[dict[str, Any]] = []
        self._collection = None

        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=DefaultEmbeddingFunction(),
            )
        except Exception:
            self._collection = None

    def add_documents(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for doc in docs:
            doc_id = doc.get("id") or str(uuid.uuid4())
            text = (doc.get("content") or "").strip()
            if not text:
                continue
            metadata = dict(doc.get("metadata") or {})
            metadata.setdefault("source", doc.get("source", "uploaded"))
            self._cache.append({"id": doc_id, "content": text, "metadata": metadata})
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)

        if self._collection and ids:
            try:
                self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            except Exception:
                pass

        self._cache = dedupe_by_key(self._cache, "id")
        return len(ids)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        if self._collection:
            try:
                result = self._collection.query(query_texts=[query], n_results=k)
                docs = []
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                ids = result.get("ids", [[]])[0]
                distances = result.get("distances", [[]])[0] if "distances" in result else [0.3] * len(documents)

                for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
                    metadata = metadata or {}
                    score = 1.0 / (1.0 + max(float(distance), 0.0))
                    docs.append(
                        {
                            "id": doc_id,
                            "content": content,
                            "source": metadata.get("source", "vector_store"),
                            "score": round(score, 3),
                            "metadata": metadata,
                        }
                    )
                if docs:
                    return docs
            except Exception:
                pass

        ranked = []
        for doc in self._cache:
            score = overlap_ratio(query, doc["content"])
            if score <= 0:
                continue
            ranked.append(
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "source": doc["metadata"].get("source", "vector_store"),
                    "score": round(score, 3),
                    "metadata": doc["metadata"],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:k]

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from src.utils.text import overlap_ratio


class SQLiteRetriever:
    def __init__(self, db_path: str = "data/structured.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.bootstrap_demo_data()

    def bootstrap_demo_data(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kpi_metrics (
                id INTEGER PRIMARY KEY,
                quarter TEXT,
                revenue_usd_mn REAL,
                gross_margin REAL,
                region TEXT
            )
            """
        )
        count = self.conn.execute("SELECT COUNT(1) AS c FROM kpi_metrics").fetchone()["c"]
        if count == 0:
            self.conn.executemany(
                "INSERT INTO kpi_metrics (quarter, revenue_usd_mn, gross_margin, region) VALUES (?, ?, ?, ?)",
                [
                    ("2025-Q1", 120.5, 0.42, "North America"),
                    ("2025-Q2", 131.2, 0.44, "Europe"),
                    ("2025-Q3", 138.9, 0.45, "Asia-Pacific"),
                    ("2025-Q4", 149.1, 0.47, "North America"),
                ],
            )
            self.conn.commit()

    def _tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return [row["name"] for row in rows]

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        scored_rows: list[dict[str, Any]] = []
        for table in self._tables():
            rows = self.conn.execute(f"SELECT * FROM {table} LIMIT 300").fetchall()
            for row in rows:
                row_dict = dict(row)
                row_text = json.dumps(row_dict, ensure_ascii=True)
                score = overlap_ratio(query, f"{table} {row_text}")
                if score <= 0:
                    continue
                scored_rows.append(
                    {
                        "id": f"sql-{table}-{row_dict.get('id', len(scored_rows))}",
                        "content": row_text,
                        "source": f"sqlite:{table}",
                        "score": round(score, 3),
                        "metadata": {"table": table},
                    }
                )

        scored_rows.sort(key=lambda item: item["score"], reverse=True)
        return scored_rows[:k]

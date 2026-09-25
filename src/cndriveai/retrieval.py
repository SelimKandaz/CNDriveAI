"""Tiny SQLite FTS5 BM25 index with explicit namespace filtering."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    namespace: str
    title: str
    body: str
    score: float


class LocalEvidenceIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(doc_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, title TEXT NOT NULL, "
                "body TEXT NOT NULL, source TEXT NOT NULL)"
            )
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING "
                "fts5(doc_id UNINDEXED, namespace UNINDEXED, title, body, tokenize='porter')"
            )

    def upsert(self, doc_id: str, namespace: str, title: str, body: str, source: str) -> None:
        if not all(value.strip() for value in (doc_id, namespace, title, body, source)):
            raise ValueError("document fields must be non-empty")
        with self._connect() as db:
            db.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
            db.execute(
                "INSERT INTO documents(doc_id, namespace, title, body, source) "
                "VALUES(?, ?, ?, ?, ?) ON CONFLICT(doc_id) DO UPDATE SET "
                "namespace=excluded.namespace, title=excluded.title, body=excluded.body, "
                "source=excluded.source",
                (doc_id, namespace, title, body, source),
            )
            db.execute(
                "INSERT INTO documents_fts(doc_id, namespace, title, body) VALUES(?, ?, ?, ?)",
                (doc_id, namespace, title, body),
            )

    def search(self, query: str, namespace: str, limit: int = 5) -> list[SearchHit]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        terms = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        with self._connect() as db:
            rows = db.execute(
                "SELECT d.doc_id, d.namespace, d.title, d.body, bm25(documents_fts) AS score "
                "FROM documents_fts f JOIN documents d ON d.doc_id=f.doc_id "
                "WHERE documents_fts MATCH ? AND d.namespace=? "
                "ORDER BY score LIMIT ?",
                (expression, namespace, limit),
            ).fetchall()
        return [
            SearchHit(row["doc_id"], row["namespace"], row["title"], row["body"], row["score"])
            for row in rows
        ]

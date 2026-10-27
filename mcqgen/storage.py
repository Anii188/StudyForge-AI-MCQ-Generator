"""Opt-in local history for generated StudyForge question sets (SQLite, no server)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parents[1] / "studyforge_history.sqlite3"


def _connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_history(db_path: str | Path = DEFAULT_DB) -> None:
    with _connect(db_path) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            title TEXT NOT NULL,
            source_preview TEXT NOT NULL,
            mode TEXT NOT NULL,
            questions_json TEXT NOT NULL,
            source_text TEXT NOT NULL
        )""")


def save_generation(title: str, source_text: str, mode: str, questions: list[dict[str, Any]], db_path: str | Path = DEFAULT_DB) -> int:
    if not questions:
        raise ValueError("Cannot save an empty question set.")
    init_history(db_path)
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO generations(created_at, title, source_preview, mode, questions_json, source_text) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                title.strip() or "Untitled study set",
                " ".join(source_text.split())[:240],
                mode,
                json.dumps(questions, ensure_ascii=False),
                source_text,
            ),
        )
        return int(cursor.lastrowid)


def list_generations(db_path: str | Path = DEFAULT_DB, limit: int = 20) -> list[dict[str, Any]]:
    init_history(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, created_at, title, source_preview, mode, questions_json FROM generations ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    return [
        {"id": row["id"], "created_at": row["created_at"], "title": row["title"],
         "source_preview": row["source_preview"], "mode": row["mode"],
         "question_count": len(json.loads(row["questions_json"]))}
        for row in rows
    ]


def load_generation(generation_id: int, db_path: str | Path = DEFAULT_DB) -> dict[str, Any] | None:
    init_history(db_path)
    with _connect(db_path) as connection:
        row = connection.execute("SELECT * FROM generations WHERE id = ?", (int(generation_id),)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "created_at": row["created_at"], "title": row["title"],
            "source_preview": row["source_preview"], "mode": row["mode"],
            "questions": json.loads(row["questions_json"]), "source_text": row["source_text"]}

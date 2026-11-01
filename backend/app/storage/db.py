"""SQLite persistence for completed experiments (metadata + compact results)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from ..config import DB_PATH_DEFAULT

_CONN_LOCK = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: str | None = None


def init(db_path: str | None = None) -> None:
    global _conn, _db_path
    path = db_path or DB_PATH_DEFAULT
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _CONN_LOCK:
        _conn = sqlite3.connect(path, check_same_thread=False)
        _db_path = path
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                results TEXT,
                error TEXT
            )
            """
        )
        _conn.commit()


def save_experiment(config: dict, results: dict | None, status: str, error: str | None = None) -> str:
    if _conn is None:
        init()
    exp_id = uuid.uuid4().hex[:12]
    with _CONN_LOCK:
        _conn.execute(
            "INSERT INTO experiments (id, created_at, status, config, results, error) VALUES (?,?,?,?,?,?)",
            (
                exp_id,
                time.time(),
                status,
                json.dumps(config),
                json.dumps(results) if results is not None else None,
                error,
            ),
        )
        _conn.commit()
    return exp_id


def get_experiment(exp_id: str) -> dict | None:
    if _conn is None:
        return None
    with _CONN_LOCK:
        row = _conn.execute(
            "SELECT id, created_at, status, config, results, error FROM experiments WHERE id=?",
            (exp_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "status": row[2],
        "config": json.loads(row[3]),
        "results": json.loads(row[4]) if row[4] else None,
        "error": row[5],
    }


def list_experiments(limit: int = 50) -> list[dict]:
    if _conn is None:
        return []
    with _CONN_LOCK:
        rows = _conn.execute(
            "SELECT id, created_at, status, config, error FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "status": r[2],
            "config": json.loads(r[3]),
            "error": r[4],
        }
        for r in rows
    ]


def delete_experiment(exp_id: str) -> bool:
    if _conn is None:
        return False
    with _CONN_LOCK:
        cur = _conn.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
        _conn.commit()
    return cur.rowcount > 0

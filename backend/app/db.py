"""SQLite persistence.

Raw sqlite3 rather than an ORM: the schema is small, the deployment target is
"one file on a laptop", and the vector column is handled explicitly anyway.
Swap `DB_URL` for PostgreSQL + pgvector when you outgrow a single node — the
access functions below are the only thing that would change.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from .config import settings

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT,
    source_kind TEXT,
    source_language TEXT DEFAULT 'auto',
    target_language TEXT DEFAULT 'es',
    duration REAL DEFAULT 0,
    sample_rate INTEGER DEFAULT 24000,
    status TEXT DEFAULT 'created',
    settings TEXT DEFAULT '{}',
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    speaker TEXT DEFAULT 'SPK_1',
    source_text TEXT DEFAULT '',
    target_text TEXT DEFAULT '',
    emotion TEXT DEFAULT 'neutral',
    voice_id TEXT,
    audio_path TEXT,
    fit_rate REAL DEFAULT 1.0,
    locked INTEGER DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id, idx);

CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,            -- preset | instant | professional | designed | converted
    category TEXT DEFAULT 'custom',
    language TEXT DEFAULT 'en',
    prompt TEXT DEFAULT '',
    params TEXT DEFAULT '{}',
    embedding BLOB,
    reference_path TEXT,
    provider TEXT DEFAULT 'local',
    provider_voice_id TEXT,
    owner TEXT DEFAULT 'local',
    tags TEXT DEFAULT '[]',
    training_seconds REAL DEFAULT 0,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_voices_owner ON voices(owner);

CREATE TABLE IF NOT EXISTS speakers (
    project_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    voice_id TEXT,
    embedding BLOB,
    total_seconds REAL DEFAULT 0,
    PRIMARY KEY (project_id, speaker)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    step TEXT DEFAULT '',
    step_index INTEGER DEFAULT 0,
    step_total INTEGER DEFAULT 0,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    error TEXT,
    params TEXT DEFAULT '{}',
    result TEXT DEFAULT '{}',
    created_at REAL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at);

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    role TEXT NOT NULL,            -- original | vocals | background | dubbed | mixed | video
    path TEXT NOT NULL,
    meta TEXT DEFAULT '{}',
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id, role);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    with tx() as conn:
        conn.executescript(SCHEMA)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def row_to_dict(row: sqlite3.Row | None, json_fields: tuple[str, ...] = ()) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for f in json_fields:
        if f in out and isinstance(out[f], str):
            try:
                out[f] = json.loads(out[f])
            except json.JSONDecodeError:
                out[f] = {}
    out.pop("embedding", None)
    return out


def pack_vector(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def search_voices_by_embedding(embedding: np.ndarray, limit: int = 10,
                               owner: str | None = None) -> list[dict[str, Any]]:
    """Cosine nearest-neighbour over the voice bank.

    A brute-force scan is the right call up to ~1e5 voices on SQLite; beyond
    that move this table to pgvector/Qdrant and keep the same signature.
    """
    query = "SELECT * FROM voices WHERE embedding IS NOT NULL"
    args: list[Any] = []
    if owner:
        query += " AND owner = ?"
        args.append(owner)
    rows = get_conn().execute(query, args).fetchall()
    if not rows:
        return []
    target = np.asarray(embedding, dtype=np.float32)
    tn = float(np.linalg.norm(target)) or 1.0
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        vec = unpack_vector(row["embedding"])
        if vec is None or vec.shape != target.shape:
            continue
        vn = float(np.linalg.norm(vec)) or 1.0
        scored.append((float(np.dot(vec, target) / (vn * tn)), row))
    scored.sort(key=lambda p: p[0], reverse=True)
    out = []
    for score, row in scored[:limit]:
        d = row_to_dict(row, ("params", "tags"))
        assert d is not None
        d["similarity"] = round(score, 4)
        out.append(d)
    return out

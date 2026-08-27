"""
SQLite-backed persistence for the API layer.

Replaces the earlier in-memory dict: without this, every restart/redeploy
silently wiped every experiment and event a team had logged -- unacceptable
for anything beyond a local demo. Kept as plain sqlite3 (no ORM) since the
schema is small and the goal is durability, not scale.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterator

# Vercel Functions have ephemeral local storage. Keeping the demo database in
# /tmp lets the read-only showcase boot successfully without pretending that
# experiment data is durable across function instances.
_default_db_path = "/tmp/a-b-testing-platform.db" if os.environ.get("VERCEL") else os.path.join("data", "platform.db")
DB_PATH = os.environ.get("AB_PLATFORM_DB_PATH", _default_db_path)

# One connection per thread, reused across calls -- FastAPI runs sync path
# operations in a thread pool, so this still gives every request its own
# connection without paying SQLite's open/close + PRAGMA setup cost on every
# single query (which dominated event-write latency: ~4ms/event opening a
# fresh connection each time vs. a small fraction of that when reused).
_local = threading.local()


def _connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # NORMAL (rather than the FULL default) skips an fsync on every commit and
    # only fsyncs at WAL checkpoints -- roughly an order of magnitude faster
    # per-event write, and still crash-safe against an application crash
    # (only an OS crash / power loss could lose the last few commits), which
    # is an acceptable trade-off for an internal tool. This combination is
    # SQLite's own recommended pairing with WAL mode.
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    return conn


@contextmanager
def _tx() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    yield conn
    conn.commit()


def init_db() -> None:
    with _tx() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                experiment_key TEXT,
                look_count INTEGER NOT NULL DEFAULT 0,
                locked INTEGER NOT NULL DEFAULT 0,
                locked_reason TEXT,
                final_result_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                user_id TEXT NOT NULL,
                grp TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_id, user_id)
            );
            """
        )
        # ALTER TABLE ADD COLUMN for a table created before experiment_key
        # existed -- CREATE TABLE IF NOT EXISTS above is a no-op on an
        # already-existing table, so this is the only thing that adds the
        # column for a DB file from before this change.
        try:
            conn.execute("ALTER TABLE experiments ADD COLUMN experiment_key TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


def get_or_create_api_key() -> str:
    """Persisted so the key survives restarts -- an API key that changes on
    every deploy would lock out every existing client."""
    with _tx() as conn:
        row = conn.execute("SELECT value FROM app_config WHERE key = 'api_key'").fetchone()
        if row:
            return row["value"]
        new_key = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO app_config (key, value) VALUES ('api_key', ?)", (new_key,))
        return new_key


def update_max_looks(experiment_id: str, new_max_looks: int) -> None:
    """Rewrites config_json's max_looks in place. Only called on a
    not-yet-locked experiment (see api/main.py's extend endpoint) -- a team
    whose experiment is still running longer than planned can raise the
    look budget instead of being forced into a premature stop_no_effect at
    the original max_looks."""
    with _tx() as conn:
        row = conn.execute("SELECT config_json FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        config_data = json.loads(row["config_json"])
        config_data["max_looks"] = new_max_looks
        conn.execute(
            "UPDATE experiments SET config_json = ? WHERE id = ?",
            (json.dumps(config_data), experiment_id),
        )


def create_experiment(experiment_id: str, name: str, config) -> str:
    """Returns the experiment's own API key (see verify_experiment_access in
    api/main.py): a team holding only this key can log events / read results
    / extend max_looks for this one experiment, without needing the global
    admin key that create_experiment itself requires."""
    experiment_key = secrets.token_urlsafe(24)
    with _tx() as conn:
        conn.execute(
            "INSERT INTO experiments (id, name, config_json, experiment_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, name, json.dumps(asdict(config)), experiment_key, datetime.now(timezone.utc).isoformat()),
        )
    return experiment_key


def get_experiment(experiment_id: str) -> sqlite3.Row | None:
    with _tx() as conn:
        return conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()


def increment_look_count(experiment_id: str) -> int:
    """Atomic within one transaction: two concurrent /results calls cannot
    observe or persist the same look number."""
    with _tx() as conn:
        conn.execute("UPDATE experiments SET look_count = look_count + 1 WHERE id = ?", (experiment_id,))
        return conn.execute(
            "SELECT look_count FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()["look_count"]


def lock_experiment(experiment_id: str, reason: str, final_result_json: str) -> None:
    with _tx() as conn:
        conn.execute(
            "UPDATE experiments SET locked = 1, locked_reason = ?, final_result_json = ? WHERE id = ?",
            (reason, final_result_json, experiment_id),
        )


def upsert_event(experiment_id: str, user_id: str, group: str, metrics: dict) -> None:
    """Last write wins per (experiment, user) -- matches real event logs
    where a user's metrics can be re-reported (e.g. a corrected conversion)."""
    with _tx() as conn:
        conn.execute(
            """
            INSERT INTO events (experiment_id, user_id, grp, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id, user_id) DO UPDATE SET
                grp = excluded.grp,
                metrics_json = excluded.metrics_json,
                created_at = excluded.created_at
            """,
            (experiment_id, user_id, group, json.dumps(metrics), datetime.now(timezone.utc).isoformat()),
        )


def get_events(experiment_id: str) -> list[dict]:
    with _tx() as conn:
        rows = conn.execute(
            "SELECT user_id, grp, metrics_json FROM events WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
    events = []
    for row in rows:
        metrics = json.loads(row["metrics_json"])
        events.append({"user_id": row["user_id"], "group": row["grp"], **metrics})
    return events

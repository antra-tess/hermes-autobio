"""Read-only data layer for the viewer.

Wraps the autobio Store with viewer-specific queries and shape
transforms. The Flask routes call into this; nothing in here should
mutate state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..store import Store, get_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def list_sessions(hermes_home: str | Path) -> List[Dict[str, Any]]:
    """Return summary stats for every session in the shared store.

    All sessions share one ``store.db`` at
    ``<hermes_home>/memory/autobio/store.db``; rows are scoped by the
    ``session_id`` column. We discover sessions by querying distinct
    session_ids across the messages table.
    """
    db = Path(hermes_home).expanduser() / "memory" / "autobio" / "store.db"
    if not db.exists():
        return []
    # We need a Store handle to query, but get_store requires a session_id.
    # Use any sentinel; we won't write through it. The lock + connection
    # are shared across get_store() calls for the same hermes_home, so
    # opening the first session_id we find via raw SQL is fine.
    import sqlite3
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    sids = set()
    try:
        for row in conn.execute(
            "SELECT DISTINCT session_id FROM messages "
            "UNION SELECT DISTINCT session_id FROM summaries"
        ):
            sids.add(row[0])
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for sid in sorted(sids):
        try:
            stats = _session_stats(str(hermes_home), sid)
        except Exception as e:
            logger.warning("viewer: skipped session %s: %s", sid, e)
            continue
        out.append({"session_id": sid, **stats})
    return out


def _session_stats(hermes_home: str, session_id: str) -> Dict[str, Any]:
    store = get_store(hermes_home, session_id)
    try:
        counts = store.summary_counts()
        return {
            "messages": store.message_count(),
            "l1": int(counts.get(1, 0)),
            "l2": int(counts.get(2, 0)),
            "l3": int(counts.get(3, 0)),
            "pins": len(store.list_pins()),
            "cm_events": len(store.list_cm_events()) if hasattr(store, "list_cm_events") else 0,
            "llm_calls": _count_llm_calls(store),
            "db_path": str(store.db_path),
        }
    finally:
        # We deliberately don't close — the singleton-cached store stays
        # warm so subsequent requests reuse it.
        pass


def _count_llm_calls(store: Store) -> int:
    try:
        with store._lock:
            row = store._conn.execute(
                "SELECT COUNT(*) AS n FROM llm_calls WHERE session_id = ?",
                (store.session_id,),
            ).fetchone()
        return int(row["n"] or 0) if row else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Message stream
# ---------------------------------------------------------------------------

def list_messages(store: Store) -> List[Dict[str, Any]]:
    """Return all archived messages in chronological order.

    Each row is annotated with ``compressed_into`` (the L1 id, if any
    — already a column on messages).
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT hash, turn, position, role, content, tool_call_id, "
            "tool_calls, tool_name, compressed_into, reasoning_details "
            "FROM messages WHERE session_id = ? "
            "ORDER BY position ASC, turn ASC",
            (store.session_id,),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                pass
        if d.get("reasoning_details"):
            try:
                d["reasoning_details"] = json.loads(d["reasoning_details"])
            except Exception:
                pass
        # Try parsing content if it looks like a list (multimodal).
        c = d.get("content")
        if isinstance(c, str) and c.startswith("["):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, list):
                    d["content"] = parsed
            except Exception:
                pass
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Summary hierarchy
# ---------------------------------------------------------------------------

def list_summaries(store: Store) -> List[Dict[str, Any]]:
    """Return all summaries (active + merged) with parsed source_hashes."""
    with store._lock:
        rows = store._conn.execute(
            "SELECT id, level, source_hashes, source_hash, content, tokens, "
            "created_at, compression_generations, pinned, merged_into, tags "
            "FROM summaries WHERE session_id = ? "
            "ORDER BY level ASC, id ASC",
            (store.session_id,),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["source_hashes"] = json.loads(d["source_hashes"])
        except Exception:
            d["source_hashes"] = []
        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except Exception:
                d["tags"] = []
        first, last = store.get_summary_turn_range(d["id"])
        d["first_turn"] = first
        d["last_turn"] = last
        out.append(d)
    return out


def build_hierarchy(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bucket summaries by level and resolve parent/child links."""
    by_level: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    by_id: Dict[str, Dict[str, Any]] = {}
    for s in summaries:
        lvl = int(s["level"])
        by_level.setdefault(lvl, []).append(s)
        by_id[s["id"]] = s
    return {
        "by_level": by_level,
        "by_id": by_id,
    }


# ---------------------------------------------------------------------------
# LLM calls + CM events
# ---------------------------------------------------------------------------

def list_llm_calls(
    store: Store, *,
    summary_id: Optional[str] = None,
    cm_event_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = store.list_llm_calls(summary_id=summary_id, cm_event_id=cm_event_id)
    return [_decorate_llm_call(r) for r in rows]


def get_llm_call(store: Store, call_id: int) -> Optional[Dict[str, Any]]:
    row = store.get_llm_call(call_id)
    return _decorate_llm_call(row) if row else None


def _decorate_llm_call(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    if d.get("prompt_json"):
        try:
            d["prompt"] = json.loads(d["prompt_json"])
        except Exception:
            d["prompt"] = []
    return d


def list_cm_events(store: Store) -> List[Dict[str, Any]]:
    rows = store.list_cm_events()
    return [_decorate_cm_event(r) for r in rows]


def get_cm_event(store: Store, event_id: int) -> Optional[Dict[str, Any]]:
    row = store.get_cm_event(event_id)
    return _decorate_cm_event(row) if row else None


def _decorate_cm_event(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    if d.get("dashboard_json"):
        try:
            d["dashboard"] = json.loads(d["dashboard_json"])
        except Exception:
            d["dashboard"] = {}
    if d.get("ops_json"):
        try:
            d["ops"] = json.loads(d["ops_json"])
        except Exception:
            d["ops"] = []
    return d


# ---------------------------------------------------------------------------
# Cross-references: which summary covers a message?
# ---------------------------------------------------------------------------

def message_to_summary_index(
    summaries: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map message hash → directly-covering L1 summary id.

    Walks L1 source_hashes only — for L2/L3 ancestry the viewer can
    follow ``merged_into`` upward.
    """
    out: Dict[str, str] = {}
    for s in summaries:
        if int(s["level"]) != 1:
            continue
        for h in s.get("source_hashes") or []:
            out[h] = s["id"]
    return out

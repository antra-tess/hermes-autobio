"""SQLite archive store for the autobiographical memory system.

Single-file schema shared by the ContextEngine and MemoryProvider halves.
Archive-first principle: every message and tool result is written here
in full *before* any pruning or compression touches the live message
list. The live list is a budget-managed view; this store is the source
of truth.

Schema summary:
  messages      — every message ever seen, keyed by content hash
  summaries     — L1/L2/L3 summaries with source_hashes chain
  pins          — user/CM pin ranges
  messages_fts  — FTS5 over message content
  summaries_fts — FTS5 over summary content + tags

The store singleton is keyed by (hermes_home, session_id) so concurrent
gateway sessions stay isolated. Both plugin halves import ``get_store``
and share the same connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level singleton: (hermes_home, session_id) -> Store.
_STORES: Dict[Tuple[str, str], "Store"] = {}
_STORES_LOCK = threading.Lock()


def get_store(hermes_home: str, session_id: str, *, subdir: str = "memory/autobio") -> "Store":
    """Return the shared Store for this (hermes_home, session_id) pair.

    Lazily opens on first call. Safe to call from both plugins — the
    second call returns the same instance.
    """
    key = (str(hermes_home), str(session_id))
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = Store(hermes_home=str(hermes_home), session_id=str(session_id), subdir=subdir)
            _STORES[key] = store
        return store


def close_store(hermes_home: str, session_id: str) -> None:
    """Close and forget the store for this session."""
    key = (str(hermes_home), str(session_id))
    with _STORES_LOCK:
        store = _STORES.pop(key, None)
    if store is not None:
        store.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    hash             TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    turn             INTEGER,
    position         INTEGER,
    role             TEXT NOT NULL,
    content          TEXT,
    tool_call_id     TEXT,
    tool_calls       TEXT,
    tool_name        TEXT,
    tokens           INTEGER,
    created_at       TEXT NOT NULL,
    compressed_into  TEXT,
    -- Provider-specific block-form metadata that must round-trip with
    -- the message body. ``reasoning_details`` holds Anthropic
    -- thinking / redacted_thinking blocks (with their signatures);
    -- Hermes' anthropic_adapter reads them back when the message is
    -- replayed to Claude. JSON-serialized list of block dicts.
    reasoning_details TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session_turn ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_messages_tool_call_id ON messages(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_messages_compressed_into ON messages(compressed_into);

CREATE TABLE IF NOT EXISTS summaries (
    id                      TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL,
    level                   INTEGER NOT NULL,
    source_hashes           TEXT NOT NULL,
    source_hash             TEXT NOT NULL,
    content                 TEXT NOT NULL,
    tokens                  INTEGER NOT NULL,
    created_at              TEXT NOT NULL,
    compression_generations INTEGER NOT NULL,
    pinned                  INTEGER NOT NULL DEFAULT 0,
    merged_into             TEXT,
    tags                    TEXT,
    first_turn              INTEGER,
    last_turn               INTEGER
);
CREATE INDEX IF NOT EXISTS idx_summaries_session_level ON summaries(session_id, level);
CREATE INDEX IF NOT EXISTS idx_summaries_merged_into ON summaries(merged_into);

-- "pins" is the table of protected ranges. A plain pin has kind='pin'
-- and keeps its range at raw L0 in the live view. A document has
-- kind='doc_chronological' or 'doc_monolithic' and carries compression
-- instructions in meta. See docs/plans/hierarchical-autobiographical-memory.md
-- "Pins and documents" for semantics.
CREATE TABLE IF NOT EXISTS pins (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    start_hash  TEXT NOT NULL,
    end_hash    TEXT NOT NULL,
    reason      TEXT,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'pin',
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_pins_session ON pins(session_id);
CREATE INDEX IF NOT EXISTS idx_pins_kind ON pins(session_id, kind);

-- Metadata for speculative compression cache and CM state.
CREATE TABLE IF NOT EXISTS kv (
    session_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);

-- DEPRECATED: speculative_cache is no longer written to. The
-- summarize/fold refactor moved dedup to summaries.source_hash lookup
-- (see Store.find_summary_by_source_hash). Table is retained without a
-- DROP to avoid migration churn on existing stores; remove in a
-- follow-up schema revision once no live stores reference it.
CREATE TABLE IF NOT EXISTS speculative_cache (
    combined_hash TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    level         INTEGER NOT NULL,
    summary       TEXT NOT NULL,
    tokens        INTEGER NOT NULL,
    source_hashes TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spec_cache_session ON speculative_cache(session_id);

-- CM events: every cm_respond / proactive_checkin call lands a row
-- here. First-class data (not just markdown report fodder), so the
-- viewer + future tooling can query without scraping. Survives
-- restarts, joinable to the LLM call that produced the reply.
CREATE TABLE IF NOT EXISTS cm_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    kind            TEXT NOT NULL,        -- 'tool_call' | 'proactive'
    turn            INTEGER,              -- engine current_turn at the time
    request         TEXT,                 -- request string ('<proactive>' for check-ins)
    dashboard_json  TEXT,                 -- JSON of the dashboard CM saw
    raw_reply       TEXT,                 -- exactly what the LLM returned
    reply_text      TEXT,                 -- parsed natural-language part
    decision        TEXT,                 -- INJECT/DEFER for proactive
    ops_json        TEXT,                 -- list of [op_line, op_result]
    llm_call_id     INTEGER               -- optional FK to llm_calls.id
);
CREATE INDEX IF NOT EXISTS idx_cm_events_session ON cm_events(session_id, created_at);

-- LLM calls: every summarizer + CM call lands a row with full prompt
-- and response. Lets the viewer show "the exact prompt that produced
-- this L1" without reconstruction guesswork. Also useful for cost
-- analysis, debugging summarizer drift, and future tuning.
CREATE TABLE IF NOT EXISTS llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    purpose             TEXT NOT NULL,    -- 'summarize' | 'merge' | 'cm' | 'proactive'
    model               TEXT,
    prompt_json         TEXT NOT NULL,    -- JSON list of message dicts (the prompt)
    response_text       TEXT,             -- the response content
    response_finish     TEXT,             -- finish_reason (stop / length / etc.)
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    summary_id          TEXT,             -- when the call produced an L1/L2/L3
    chunk_source_hash   TEXT,             -- when the call was for a specific chunk
    cm_event_id         INTEGER,          -- when the call drove a CM event
    error               TEXT              -- if the call failed
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_summary ON llm_calls(summary_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_cm ON llm_calls(cm_event_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    hash UNINDEXED,
    session_id UNINDEXED,
    role UNINDEXED,
    content,
    tokenize = 'porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    id UNINDEXED,
    session_id UNINDEXED,
    level UNINDEXED,
    content,
    tags,
    tokenize = 'porter unicode61'
);
"""


class Store:
    """SQLite-backed archive. Thread-safe via a coarse lock."""

    def __init__(self, hermes_home: str, session_id: str, subdir: str = "memory/autobio"):
        self.session_id = session_id
        self.hermes_home = Path(hermes_home)
        self.dir = self.hermes_home / subdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "store.db"
        # Summary markdown files live alongside the DB for grep-ability.
        self.summaries_dir = self.dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)

        self._lock = threading.RLock()
        # check_same_thread=False + our own lock: the CM loop and the main
        # turn may interleave writes from different threads.
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit transactions via BEGIN
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Idempotent in-place schema bumps for older stores.

        SQLite's ``CREATE TABLE IF NOT EXISTS`` never alters an
        existing table, so additive column changes need explicit
        ``ALTER TABLE`` guarded by an existence check. Each migration
        here is safe to re-run on an up-to-date schema.
        """
        with self._lock:
            pin_cols = {
                row["name"] for row in self._conn.execute(
                    "PRAGMA table_info(pins)"
                ).fetchall()
            }
            if "kind" not in pin_cols:
                self._conn.execute(
                    "ALTER TABLE pins ADD COLUMN kind TEXT NOT NULL DEFAULT 'pin'"
                )
                logger.info("autobio: migrated pins table (added kind column)")
            if "meta" not in pin_cols:
                self._conn.execute("ALTER TABLE pins ADD COLUMN meta TEXT")
                logger.info("autobio: migrated pins table (added meta column)")
            # Ensure the new index exists (CREATE INDEX IF NOT EXISTS
            # handles it, but we only ran SCHEMA once, so re-assert).
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pins_kind ON pins(session_id, kind)"
            )
            msg_cols = {
                row["name"] for row in self._conn.execute(
                    "PRAGMA table_info(messages)"
                ).fetchall()
            }
            if "reasoning_details" not in msg_cols:
                self._conn.execute(
                    "ALTER TABLE messages ADD COLUMN reasoning_details TEXT"
                )
                logger.info(
                    "autobio: migrated messages table "
                    "(added reasoning_details column)"
                )

            # Tool-call shape migration: older stores flatten tool_calls
            # to ``{id, name, arguments}``. The current schema stores
            # canonical OpenAI form ``{id, type, function: {name,
            # arguments}}`` so every code path sees the same shape and
            # content hashes agree. Re-inflate any flat-form rows.
            #
            # Idempotent: rows already in canonical form are detected by
            # the ``function`` key and skipped. Safe to re-run on every
            # open. Cheap for stores whose rows are already migrated
            # (one SELECT, no UPDATEs).
            try:
                rows = self._conn.execute(
                    "SELECT hash, tool_calls FROM messages "
                    "WHERE session_id = ? AND tool_calls IS NOT NULL",
                    (self.session_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            migrated = 0
            for row in rows:
                raw = row["tool_calls"]
                try:
                    parsed = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(parsed, list) or not parsed:
                    continue
                first = parsed[0]
                if (
                    isinstance(first, dict)
                    and isinstance(first.get("function"), dict)
                ):
                    continue   # already canonical
                # Re-inflate.
                canonical = []
                for tc in parsed:
                    if not isinstance(tc, dict):
                        continue
                    canonical.append({
                        "id": tc.get("id", "") or "",
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "") or "",
                            "arguments": tc.get("arguments", "") or "",
                        },
                    })
                self._conn.execute(
                    "UPDATE messages SET tool_calls = ? WHERE hash = ?",
                    (json.dumps(canonical), row["hash"]),
                )
                migrated += 1
            if migrated:
                logger.info(
                    "autobio: migrated %d messages.tool_calls rows "
                    "to canonical OpenAI shape",
                    migrated,
                )

            # CM events + LLM call provenance tables. Idempotent —
            # CREATE TABLE IF NOT EXISTS handles fresh stores; this is
            # for older stores that pre-date these tables.
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS cm_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    kind            TEXT NOT NULL,
                    turn            INTEGER,
                    request         TEXT,
                    dashboard_json  TEXT,
                    raw_reply       TEXT,
                    reply_text      TEXT,
                    decision        TEXT,
                    ops_json        TEXT,
                    llm_call_id     INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_cm_events_session
                    ON cm_events(session_id, created_at);
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id          TEXT NOT NULL,
                    created_at          TEXT NOT NULL,
                    purpose             TEXT NOT NULL,
                    model               TEXT,
                    prompt_json         TEXT NOT NULL,
                    response_text       TEXT,
                    response_finish     TEXT,
                    prompt_tokens       INTEGER,
                    completion_tokens   INTEGER,
                    summary_id          TEXT,
                    chunk_source_hash   TEXT,
                    cm_event_id         INTEGER,
                    error               TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_llm_calls_session
                    ON llm_calls(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_summary
                    ON llm_calls(summary_id);
                CREATE INDEX IF NOT EXISTS idx_llm_calls_cm
                    ON llm_calls(cm_event_id);
            """)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception as e:
                # Don't suppress: a close error can mean WAL not flushed,
                # pending transaction, or filesystem trouble — all of
                # which can lose data. Log loudly so the human notices
                # before the next session starts.
                logger.error(
                    "autobio store.close FAILED for session=%s db=%s: %s",
                    self.session_id, self.db_path, e,
                )

    # -- messages --------------------------------------------------------

    def upsert_message(
        self,
        hash_id: str,
        message: Dict[str, Any],
        *,
        turn: Optional[int] = None,
        position: Optional[int] = None,
        tokens: Optional[int] = None,
    ) -> None:
        """Insert or ignore a message by hash.

        Idempotent — if the hash exists, this is a no-op (turn/position
        from the first insert win, which is the correct behavior for
        archive-first: the *original* position is what matters).
        """
        tool_calls = message.get("tool_calls")
        tool_calls_json = (
            json.dumps(_canonicalize_tool_calls(tool_calls))
            if tool_calls else None
        )
        content = message.get("content")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        # Provider-preserved fields that must round-trip with the
        # message — currently only Anthropic thinking blocks. JSON
        # serialized so the column can hold a list of dicts.
        reasoning_details = message.get("reasoning_details")
        reasoning_json = (
            json.dumps(reasoning_details, ensure_ascii=False)
            if reasoning_details else None
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO messages (
                    hash, session_id, turn, position, role, content,
                    tool_call_id, tool_calls, tool_name, tokens, created_at,
                    reasoning_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hash_id,
                    self.session_id,
                    turn,
                    position,
                    message.get("role", ""),
                    content,
                    message.get("tool_call_id"),
                    tool_calls_json,
                    message.get("name"),
                    tokens,
                    _now(),
                    reasoning_json,
                ),
            )
            # Mirror into FTS. OR IGNORE upstream means we only want
            # to insert into FTS when the messages row was freshly added —
            # check via rowcount.
            cur = self._conn.execute("SELECT changes() AS n")
            if (cur.fetchone()["n"] or 0) > 0:
                fts_content = content if isinstance(content, str) else (content or "")
                self._conn.execute(
                    "INSERT INTO messages_fts (hash, session_id, role, content) VALUES (?, ?, ?, ?)",
                    (hash_id, self.session_id, message.get("role", ""), fts_content),
                )

    def get_message(self, hash_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE hash = ?", (hash_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_messages_by_hashes(self, hashes: List[str]) -> List[Dict[str, Any]]:
        """Return archived messages for a list of hashes, preserving order."""
        if not hashes:
            return []
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE hash IN ({placeholders})", hashes
            ).fetchall()
        by_hash = {row["hash"]: dict(row) for row in rows}
        return [by_hash[h] for h in hashes if h in by_hash]

    def get_tool_result(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        """Look up the full archived tool result for a call ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE role = 'tool' AND tool_call_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (tool_call_id,),
            ).fetchone()
        return dict(row) if row else None

    def mark_compressed(self, hashes: List[str], summary_id: str) -> None:
        """Mark messages as having been compressed into ``summary_id``.

        The rows remain in the archive — this only records which summary
        they were rolled into. Used for uncompression (finding the
        originals behind an L1).

        Raises ``ValueError`` if ``summary_id`` does not exist or is
        already marked merged_into another. Without this check we
        could leave messages pointing at a phantom or inactive summary,
        breaking the uncompression chain silently.
        """
        if not hashes:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT id, merged_into FROM summaries "
                "WHERE id = ? AND session_id = ?",
                (summary_id, self.session_id),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"mark_compressed: summary {summary_id!r} not in archive — "
                    f"refusing to leave dangling pointers on {len(hashes)} messages"
                )
            if row["merged_into"] is not None:
                raise ValueError(
                    f"mark_compressed: summary {summary_id!r} is merged_into "
                    f"{row['merged_into']!r} — pointing messages at an inactive "
                    f"summary breaks unfold"
                )
            placeholders = ",".join("?" * len(hashes))
            self._conn.execute(
                f"UPDATE messages SET compressed_into = ? WHERE hash IN ({placeholders})",
                [summary_id, *hashes],
            )

    # -- summaries -------------------------------------------------------

    def next_summary_id(self, level: int) -> str:
        """Return the next free summary ID for a level (e.g. 'L1-0014')."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM summaries WHERE session_id = ? AND level = ?",
                (self.session_id, level),
            ).fetchone()
        n = (row["n"] or 0) + 1
        return f"L{level}-{n:04d}"

    def insert_summary(
        self,
        *,
        level: int,
        source_hashes: List[str],
        source_hash: str,
        content: str,
        tokens: int,
        compression_generations: int,
        tags: Optional[List[str]] = None,
        # first_turn/last_turn are NOT stored. Turn ranges are derived
        # on demand from source_hashes via get_summary_turn_range —
        # storing them created a stale label problem and turn numbers
        # are positional, not identity. The DB columns remain for
        # schema stability but are always NULL.
        **_deprecated,
    ) -> str:
        """Insert a new summary and return its assigned ID."""
        tags_json = json.dumps(tags) if tags else None
        source_hashes_json = json.dumps(source_hashes)
        now = _now()

        with self._lock:
            # Dedup under lock: two threads (main compress + speculative
            # worker) can both pass the outer find_summary_by_source_hash
            # check, both make an LLM call, then race into insert. Check
            # again here inside the lock; if a row already exists for
            # this (session, level, source_hash), return the winner's id
            # and discard the losing LLM output.
            #
            # Note: this query does NOT filter ``merged_into IS NULL``
            # because the goal is to prevent any duplicate write of the
            # same source_hash. If the existing summary has been merged
            # into a parent, returning its id is still correct — the
            # caller (summarize_chunk → mark_compressed) will validate
            # before writing compressed_into. ``find_summary_by_source_hash``
            # uses the ``merged_into IS NULL`` filter for a different
            # reason (return only active summaries the caller can reuse
            # in the live view), and that's the right contract there.
            if source_hash:
                row = self._conn.execute(
                    "SELECT id FROM summaries "
                    "WHERE session_id = ? AND level = ? AND source_hash = ? "
                    "LIMIT 1",
                    (self.session_id, level, source_hash),
                ).fetchone()
                if row is not None:
                    return row[0]
            summary_id = self.next_summary_id(level)
            self._conn.execute(
                """
                INSERT INTO summaries (
                    id, session_id, level, source_hashes, source_hash, content,
                    tokens, created_at, compression_generations, pinned,
                    merged_into, tags, first_turn, last_turn
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL, NULL)
                """,
                (
                    summary_id, self.session_id, level, source_hashes_json,
                    source_hash, content, tokens, now, compression_generations,
                    tags_json,
                ),
            )
            self._conn.execute(
                "INSERT INTO summaries_fts (id, session_id, level, content, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (summary_id, self.session_id, level, content, tags_json or ""),
            )

        # Write the markdown file outside the lock (slow filesystem op).
        md_dir = self.summaries_dir / f"L{level}"
        md_dir.mkdir(exist_ok=True)
        md_path = md_dir / f"{summary_id}.md"
        md_path.write_text(
            f"<!-- {summary_id} session={self.session_id} "
            f"gen={compression_generations} -->\n\n"
            + content,
            encoding="utf-8",
        )
        return summary_id

    def get_summary_position_range(
        self, summary_id: str,
    ) -> "tuple[Optional[int], Optional[int]]":
        """Derive ``(min_position, max_position)`` from the messages or
        child summaries this summary covers.

        Mirrors ``get_summary_turn_range`` but works in archive-position
        space, which is the coordinate system the merge-contiguity check
        cares about. ``turn`` numbers are also positional but they can
        be ``NULL`` for messages archived before the turn counter was
        wired up; ``position`` is set on every row.

        Returns ``(None, None)`` if no source resolves — orphan
        source_hashes, or a child summary that was never written.
        """
        summary = self.get_summary(summary_id)
        if summary is None:
            return (None, None)
        try:
            source_hashes = json.loads(summary.get("source_hashes") or "[]")
        except Exception:
            return (None, None)
        if not source_hashes:
            return (None, None)
        level = int(summary.get("level") or 0)
        if level == 1:
            placeholders = ",".join("?" * len(source_hashes))
            with self._lock:
                row = self._conn.execute(
                    f"SELECT MIN(position), MAX(position) FROM messages "
                    f"WHERE session_id = ? AND hash IN ({placeholders})",
                    (self.session_id, *source_hashes),
                ).fetchone()
            if row is None or row[0] is None:
                return (None, None)
            return (row[0], row[1])
        # L2+: recurse over child summaries.
        firsts: List[int] = []
        lasts: List[int] = []
        for child_id in source_hashes:
            f, l = self.get_summary_position_range(child_id)
            if f is not None:
                firsts.append(f)
            if l is not None:
                lasts.append(l)
        return (
            min(firsts) if firsts else None,
            max(lasts) if lasts else None,
        )

    def get_summary_turn_range(
        self, summary_id: str,
    ) -> "tuple[Optional[int], Optional[int]]":
        """Derive ``(first_turn, last_turn)`` on demand from source_hashes.

        For L1, source_hashes are message content-hashes; we look each
        one up in ``messages`` and return MIN/MAX of the ``turn``
        column. For L2/L3, source_hashes are child summary IDs; we
        recurse and take the extremes across children.

        Returns ``(None, None)`` if nothing resolves — e.g., a summary
        references messages that were never archived, or whose rows
        have since been removed. Never returns stale cached values:
        this is the point of deriving on demand rather than storing.
        """
        summary = self.get_summary(summary_id)
        if summary is None:
            return (None, None)
        try:
            source_hashes = json.loads(summary.get("source_hashes") or "[]")
        except Exception:
            return (None, None)
        if not source_hashes:
            return (None, None)
        level = int(summary.get("level") or 0)
        if level == 1:
            placeholders = ",".join("?" * len(source_hashes))
            with self._lock:
                row = self._conn.execute(
                    f"SELECT MIN(turn), MAX(turn) FROM messages "
                    f"WHERE session_id = ? AND hash IN ({placeholders})",
                    (self.session_id, *source_hashes),
                ).fetchone()
            if row is None:
                return (None, None)
            return (row[0], row[1])
        # L2+: recurse over child summaries.
        firsts: List[int] = []
        lasts: List[int] = []
        for child_id in source_hashes:
            f, l = self.get_summary_turn_range(child_id)
            if f is not None:
                firsts.append(f)
            if l is not None:
                lasts.append(l)
        return (
            min(firsts) if firsts else None,
            max(lasts) if lasts else None,
        )

    def find_summary_by_source_hash(
        self, source_hash: str, level: int,
    ) -> Optional[Dict[str, Any]]:
        """Return an existing active summary matching this source_hash, if any.

        Used to dedup summarization work: if a chunk's combined source_hash
        matches an already-committed L1, there is no need to call the LLM
        again. The archive IS the cache — once a chunk has been summarized,
        that memory is authoritative regardless of what later context
        changes might occur.

        Only returns non-merged summaries (active ones). A merged child
        has been superseded by its parent and should not be reused.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM summaries "
                "WHERE session_id = ? AND level = ? AND source_hash = ? "
                "AND merged_into IS NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (self.session_id, level, source_hash),
            ).fetchone()
        return dict(row) if row else None

    def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM summaries WHERE id = ?", (summary_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_summaries(
        self, level: int, *, include_merged: bool = False
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT * FROM summaries WHERE session_id = ? AND level = ?"
            + ("" if include_merged else " AND merged_into IS NULL")
            + " ORDER BY created_at ASC"
        )
        with self._lock:
            rows = self._conn.execute(sql, (self.session_id, level)).fetchall()
        return [dict(r) for r in rows]

    def mark_summary_merged(self, summary_ids: List[str], merged_into: str) -> None:
        if not summary_ids:
            return
        placeholders = ",".join("?" * len(summary_ids))
        with self._lock:
            self._conn.execute(
                f"UPDATE summaries SET merged_into = ? WHERE id IN ({placeholders})",
                [merged_into, *summary_ids],
            )

    def set_pin(self, summary_id: str, pinned: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE summaries SET pinned = ? WHERE id = ?",
                (1 if pinned else 0, summary_id),
            )

    # -- pins / protected ranges ----------------------------------------
    #
    # A "pin" is the umbrella concept: a range of messages the live view
    # should keep at a specific resolution. kind discriminates behavior:
    #
    #   'pin'                 plain pin — keep raw, don't fold until unpin
    #   'doc_chronological'   document, fold-from-top under pressure
    #   'doc_monolithic'      document, fold whole or not at all
    #
    # The meta column (JSON) carries kind-specific attributes:
    # original_tokens, title, auto_source, etc.

    def add_pin(
        self,
        start_hash: str,
        end_hash: str,
        reason: str = "",
        *,
        kind: str = "pin",
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Register a protected range. Returns the new pin's id."""
        meta_json = json.dumps(meta) if meta else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO pins "
                "(session_id, start_hash, end_hash, reason, created_at, kind, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, start_hash, end_hash, reason,
                 _now(), kind, meta_json),
            )
            return cur.lastrowid

    def list_pins(self, *, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return active pins for this session, optionally filtered by kind.

        Parses the ``meta`` JSON into a dict on each row for convenience.
        """
        sql = "SELECT * FROM pins WHERE session_id = ?"
        params: List[Any] = [self.session_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            raw_meta = d.get("meta")
            if isinstance(raw_meta, str):
                try:
                    d["meta"] = json.loads(raw_meta)
                except json.JSONDecodeError:
                    d["meta"] = None
            out.append(d)
        return out

    def advance_pin_start(self, pin_id: int, new_start_hash: str) -> None:
        """Move a pin's start_hash forward. Used when a doc's top chunk folds."""
        with self._lock:
            self._conn.execute(
                "UPDATE pins SET start_hash = ? WHERE id = ?",
                (new_start_hash, pin_id),
            )

    def remove_pin(self, pin_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pins WHERE id = ?", (pin_id,))

    def find_ranges_in_view(
        self, live_hashes: List[str], *, kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Locate active pins whose start AND end hashes both appear in a
        given live message list, and return their positions.

        ``live_hashes`` is the ordered list of content hashes of the
        current live messages. A pin only "applies" to the chunk picker
        if both endpoints are visible — if a pin's content has been
        fully folded away already, it's not in the live view and the
        picker can ignore it.

        Returns a list of dicts with: pin_id, kind, meta, start_index,
        end_index (inclusive), reason. Missing-in-view pins are skipped.
        """
        pins = self.list_pins(kind=kind)
        if not pins:
            return []
        hash_index: Dict[str, int] = {}
        for i, h in enumerate(live_hashes):
            # First occurrence wins — message hashes shouldn't collide
            # in normal use, but if they do we prefer the earlier index.
            hash_index.setdefault(h, i)

        results: List[Dict[str, Any]] = []
        for p in pins:
            s = hash_index.get(p["start_hash"])
            e = hash_index.get(p["end_hash"])
            if s is None or e is None or e < s:
                continue
            results.append({
                "pin_id": p["id"],
                "kind": p["kind"],
                "meta": p.get("meta"),
                "reason": p.get("reason"),
                "start_hash": p["start_hash"],
                "end_hash": p["end_hash"],
                "start_index": s,
                "end_index": e,
            })
        return results

    # -- speculative cache (DEPRECATED) ----------------------------------
    # These methods are unused since the summarize/fold refactor moved
    # dedup into summaries.source_hash lookup. Kept for the moment so
    # schema-compatibility tests on older stores don't break. Slated
    # for removal along with the speculative_cache table.

    def cache_speculative(
        self,
        combined_hash: str,
        *,
        level: int,
        summary: str,
        tokens: int,
        source_hashes: List[str],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO speculative_cache "
                "(combined_hash, session_id, level, summary, tokens, source_hashes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    combined_hash, self.session_id, level, summary, tokens,
                    json.dumps(source_hashes), _now(),
                ),
            )

    def get_speculative(self, combined_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speculative_cache WHERE combined_hash = ?",
                (combined_hash,),
            ).fetchone()
        return dict(row) if row else None

    def purge_speculative(self) -> int:
        """Drop cached speculative results for this session (e.g. on rollback)."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM speculative_cache WHERE session_id = ?", (self.session_id,)
            )
            return cur.rowcount or 0

    # -- KV state --------------------------------------------------------

    def kv_set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (session_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, key, json.dumps(value), _now()),
            )

    def kv_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE session_id = ? AND key = ?",
                (self.session_id, key),
            ).fetchone()
        if not row or row["value"] is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    # -- search ----------------------------------------------------------

    def fts_messages(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """BM25 search over archived messages.

        Returns ``[]`` on FTS errors (malformed query OR corrupt index)
        but logs at error level — callers can't distinguish "no hits"
        from "broken FTS" from the return value, so the log is the
        operator's signal.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT m.hash, m.role, m.turn, m.content, bm25(messages_fts) AS score "
                    "FROM messages_fts JOIN messages m ON m.hash = messages_fts.hash "
                    "WHERE messages_fts MATCH ? AND m.session_id = ? "
                    "ORDER BY score ASC LIMIT ?",
                    (query, self.session_id, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.error(
                    "autobio FTS messages query FAILED (returning []): "
                    "session=%s query=%r error=%s",
                    self.session_id, query, e,
                )
                return []
        return [dict(r) for r in rows]

    def fts_summaries(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """BM25 search over summary content + tags.

        Turn range is not returned here — callers who need it must use
        ``get_summary_turn_range(id)``. Storing a pre-computed range on
        the summary led to stale labels; derive on demand.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT s.id, s.level, s.compression_generations, "
                    "s.content, bm25(summaries_fts) AS score "
                    "FROM summaries_fts JOIN summaries s ON s.id = summaries_fts.id "
                    "WHERE summaries_fts MATCH ? AND s.session_id = ? "
                    "AND s.merged_into IS NULL "
                    "ORDER BY score ASC LIMIT ?",
                    (query, self.session_id, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.error(
                    "autobio FTS summaries query FAILED (returning []): "
                    "session=%s query=%r error=%s",
                    self.session_id, query, e,
                )
                return []
        return [dict(r) for r in rows]

    # -- stats -----------------------------------------------------------

    def summary_counts(self) -> Dict[int, int]:
        """Return {level: count_of_active_summaries} for the CM dashboard."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT level, COUNT(*) AS n FROM summaries "
                "WHERE session_id = ? AND merged_into IS NULL GROUP BY level",
                (self.session_id,),
            ).fetchall()
        return {int(r["level"]): int(r["n"]) for r in rows}

    def message_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        return int(row["n"] or 0)

    # -- CM events + LLM call provenance ---------------------------------

    def record_llm_call(
        self,
        *,
        purpose: str,
        prompt_messages: List[Dict[str, Any]],
        response_text: Optional[str],
        model: Optional[str] = None,
        response_finish: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        summary_id: Optional[str] = None,
        chunk_source_hash: Optional[str] = None,
        cm_event_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> int:
        """Record one LLM call. Returns the new row id.

        ``prompt_messages`` is stored as JSON — the full prompt that
        went to the model, in OpenAI message-list shape. Lets the
        viewer render the exact request as a turn-by-turn conversation
        rather than reconstructing it from the archive.
        """
        prompt_json = json.dumps(prompt_messages, ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO llm_calls (
                    session_id, created_at, purpose, model, prompt_json,
                    response_text, response_finish, prompt_tokens,
                    completion_tokens, summary_id, chunk_source_hash,
                    cm_event_id, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, _now(), purpose, model, prompt_json,
                    response_text, response_finish, prompt_tokens,
                    completion_tokens, summary_id, chunk_source_hash,
                    cm_event_id, error,
                ),
            )
            return int(cur.lastrowid)

    def attach_summary_to_llm_call(self, call_id: int, summary_id: str) -> None:
        """Backfill the summary_id on an LLM call after the summary
        is written. The recorder doesn't know the summary id at call
        time (it's assigned during insert); this links them after.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE llm_calls SET summary_id = ? WHERE id = ?",
                (summary_id, call_id),
            )

    def attach_cm_event_to_llm_call(self, call_id: int, cm_event_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE llm_calls SET cm_event_id = ? WHERE id = ?",
                (cm_event_id, call_id),
            )

    def record_cm_event(
        self,
        *,
        kind: str,
        request: str,
        dashboard: Dict[str, Any],
        raw_reply: str,
        reply_text: str,
        ops: List["tuple[str, str]"],
        decision: Optional[str] = None,
        turn: Optional[int] = None,
        llm_call_id: Optional[int] = None,
    ) -> int:
        """Record one CM dialogue event. Returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO cm_events (
                    session_id, created_at, kind, turn, request,
                    dashboard_json, raw_reply, reply_text, decision,
                    ops_json, llm_call_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, _now(), kind, turn, request,
                    json.dumps(dashboard, ensure_ascii=False),
                    raw_reply, reply_text, decision,
                    json.dumps(ops, ensure_ascii=False),
                    llm_call_id,
                ),
            )
            return int(cur.lastrowid)

    def list_cm_events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = ("SELECT * FROM cm_events WHERE session_id = ? "
             "ORDER BY id ASC")
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(q, (self.session_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_cm_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cm_events WHERE id = ?", (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_llm_calls(
        self, *, summary_id: Optional[str] = None,
        cm_event_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses = ["session_id = ?"]
        params: List[Any] = [self.session_id]
        if summary_id is not None:
            clauses.append("summary_id = ?")
            params.append(summary_id)
        if cm_event_id is not None:
            clauses.append("cm_event_id = ?")
            params.append(cm_event_id)
        q = (f"SELECT * FROM llm_calls WHERE {' AND '.join(clauses)} "
             f"ORDER BY id ASC")
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def get_llm_call(self, call_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM llm_calls WHERE id = ?", (call_id,),
            ).fetchone()
        return dict(row) if row else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonicalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Produce canonical OpenAI tool-call shape for storage.

    Output: ``[{id, type:"function", function:{name, arguments}}]``.

    Earlier revisions of this module flattened tool_calls to the form
    ``{id, name, arguments}`` and the render layer re-inflated on the
    way out. That round-trip was a footgun: any code path that bypassed
    the inflation step (the picker over a synthetic message dict, a
    validator reading a stored row, etc.) saw a different shape than
    the harness emits, hashed differently, and produced duplicate
    archive rows for the same tool call. We now store the canonical
    form verbatim — no flatten on insert, no inflation on read.

    Inputs we still have to normalize:

    - **dict, canonical**: pass-through (with type defaulted to
      ``"function"`` since OpenAI clients sometimes omit it).
    - **dict, flat (legacy)**: re-inflate to canonical (defensive — by
      the time this runs the migration has already canonicalized stored
      rows, but a flat dict could still arrive from a misconfigured
      caller).
    - **SimpleNamespace** (from the SDK's ChatCompletion objects):
      lift attributes into a canonical dict so the result is JSON
      serializable.
    """
    out = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                # Canonical form: keep it.
                out.append({
                    "id": tc.get("id", "") or "",
                    "type": tc.get("type", "function") or "function",
                    "function": {
                        "name": fn.get("name", "") or "",
                        "arguments": fn.get("arguments", "") or "",
                    },
                })
            else:
                # Flat form ``{id, name, arguments}`` — re-inflate.
                out.append({
                    "id": tc.get("id", "") or "",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", "") or "",
                        "arguments": tc.get("arguments", "") or "",
                    },
                })
        else:
            fn = getattr(tc, "function", None)
            out.append({
                "id": getattr(tc, "id", "") or "",
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": (getattr(fn, "name", "") or "") if fn else "",
                    "arguments": (getattr(fn, "arguments", "") or "") if fn else "",
                },
            })
    return out

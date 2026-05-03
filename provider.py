"""AutobioMemoryProvider — the search + tool-exposure half of the plugin.

V1 skeleton: opens the shared store on initialize, tracks turn count,
and is ready to host ``memory_search`` / ``memory_grep`` / ``memory_read``
tools once task 9 lands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .config import load_autobio_config
from .store import Store, close_store, get_store

logger = logging.getLogger(__name__)


class AutobioMemoryProvider(MemoryProvider):
    """Hierarchical autobiographical memory — search + provider side."""

    def __init__(self) -> None:
        self.cfg = load_autobio_config()
        self._session_id: Optional[str] = None
        self._hermes_home: Optional[str] = None
        self._store: Optional[Store] = None

    @property
    def name(self) -> str:
        return "autobio"

    def is_available(self) -> bool:
        # Local SQLite — no external deps. Always available when the
        # plugin is selected.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
        self._store = get_store(
            self._hermes_home, session_id,
            subdir=self.cfg["store"]["subdir"],
        )
        logger.info("autobio provider: session=%s store=%s", session_id, self._store.db_path)

    def shutdown(self) -> None:
        if self._hermes_home and self._session_id:
            close_store(self._hermes_home, self._session_id)
        self._store = None

    # -- turn tracking (shared with the engine via the store's KV) -----

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # Track the turn counter in shared state — the engine reads this
        # when building its dashboard and scheduling check-ins. The
        # check-in itself runs from the engine's on_turn_end hook, which
        # has real access to the live message list.
        if self._store:
            self._store.kv_set("current_turn", turn_number)
            since = int(self._store.kv_get("turns_since_checkin", 0) or 0) + 1
            self._store.kv_set("turns_since_checkin", since)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # FTS indexing happens automatically when messages enter the
        # archive via the engine's compress() pass. Nothing to do here
        # for v1.
        pass

    # -- tools ---------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": (
                    "BM25 search across your autobiographical memory — all "
                    "compressed summaries (L1/L2/L3) plus recent archived "
                    "messages. Use this for finding prior decisions, topics, "
                    "discussions by keyword or phrase. Returns up to "
                    "max_results hits with provenance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms."},
                        "max_results": {"type": "integer", "description": "Result cap (default 6)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_grep",
                "description": (
                    "Raw text search across the on-disk summary archive. "
                    "Use for exact matches that BM25 might miss — function "
                    "names, error codes, specific values, URLs, file paths."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern."},
                        "max_results": {"type": "integer", "description": "Result cap (default 20)."},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "memory_read",
                "description": (
                    "Read the full content of a specific summary (L1/L2/L3) "
                    "or an archived message by its hash. Use after "
                    "memory_search surfaces a relevant source."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Summary ID (e.g. L1-0014) or 16-char message hash.",
                        },
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "mark_document",
                "description": (
                    "Mark a range of messages as a 'document' — a "
                    "protected region the live view should keep at raw "
                    "resolution until memory pressure forces a fold. "
                    "Use after reading a large tool result (long log, "
                    "transcript, report) that you expect to need in "
                    "full for a while.\n\n"
                    "Specify the range in one of three ways:\n"
                    "  - tool_call_id: mark the single tool result "
                    "with this call_id\n"
                    "  - start_turn + end_turn: mark messages between "
                    "these turns (inclusive)\n"
                    "  - start_hash + end_hash: precise hash-based range\n\n"
                    "Documents fold from the top (oldest end first) "
                    "when pressure demands it. Chronological style is "
                    "the default and works for logs, transcripts, and "
                    "timestamped data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_call_id": {
                            "type": "string",
                            "description": "Mark a tool result by its call_id (single-message document).",
                        },
                        "start_turn": {
                            "type": "integer",
                            "description": "First turn in the document range (inclusive).",
                        },
                        "end_turn": {
                            "type": "integer",
                            "description": "Last turn in the document range (inclusive).",
                        },
                        "start_hash": {
                            "type": "string",
                            "description": "First message content hash.",
                        },
                        "end_hash": {
                            "type": "string",
                            "description": "Last message content hash.",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["chronological", "monolithic"],
                            "description": "Fold strategy. Default: chronological.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional human-readable title for the document.",
                        },
                    },
                    "required": [],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        import json
        if self._store is None:
            return json.dumps({"error": "autobio provider: store not open"})

        from .search import expand_summary, grep_archive, search_all

        if tool_name == "memory_search":
            query = (args or {}).get("query", "")
            max_results = int((args or {}).get("max_results", 6))
            if not query:
                return json.dumps({"error": "memory_search requires 'query'"})
            hits = search_all(self._store, query, max_results=max_results)
            return json.dumps({"query": query, "results": hits})

        if tool_name == "memory_grep":
            pattern = (args or {}).get("pattern", "")
            max_results = int((args or {}).get("max_results", 20))
            if not pattern:
                return json.dumps({"error": "memory_grep requires 'pattern'"})
            try:
                hits = grep_archive(self._store, pattern, limit=max_results)
            except ValueError as e:
                return json.dumps({"error": str(e)})
            return json.dumps({"pattern": pattern, "results": hits})

        if tool_name == "mark_document":
            return self._handle_mark_document(args or {})

        if tool_name == "memory_read":
            ident = (args or {}).get("id", "")
            if not ident:
                return json.dumps({"error": "memory_read requires 'id'"})
            # Summary IDs look like L1-0014, L2-0003, L3-0001.
            if ident.startswith(("L1-", "L2-", "L3-")):
                s = self._store.get_summary(ident)
                if not s:
                    return json.dumps({"error": f"summary {ident} not found"})
                first, last = self._store.get_summary_turn_range(ident)
                from .search import UnfoldFailed
                children = None
                children_error: Optional[str] = None
                if s["level"] > 1:
                    try:
                        children = expand_summary(self._store, ident)
                    except UnfoldFailed as fail:
                        children_error = str(fail)
                return json.dumps({
                    "id": s["id"], "level": s["level"],
                    "compression_generations": s["compression_generations"],
                    "first_turn": first, "last_turn": last,
                    "content": s["content"],
                    "children": children,
                    "children_error": children_error,
                })
            # Otherwise treat as a message hash.
            m = self._store.get_message(ident)
            if not m:
                return json.dumps({"error": f"message {ident} not found"})
            return json.dumps({
                "hash": m["hash"], "role": m["role"], "turn": m["turn"],
                "content": m.get("content"),
                "tool_call_id": m.get("tool_call_id"),
                "compressed_into": m.get("compressed_into"),
            })

        return json.dumps({"error": f"autobio provider: unknown tool {tool_name}"})

    # -- mark_document handler -----------------------------------------

    def _handle_mark_document(self, args: Dict[str, Any]) -> str:
        """Resolve the target range and register a document pin.

        Three resolution paths, in priority order:
          1. tool_call_id → single-message doc
          2. start_hash + end_hash → explicit range
          3. start_turn + end_turn → turn-inclusive range

        Returns JSON with {pin_id, kind, start_hash, end_hash} on
        success, or {error: "..."} otherwise.
        """
        import json as _json

        style = args.get("style", "chronological")
        if style not in ("chronological", "monolithic"):
            return _json.dumps({"error": f"invalid style: {style}"})
        kind = f"doc_{style}"

        title = args.get("title")

        start_hash, end_hash = self._resolve_doc_range(args)
        if isinstance(start_hash, str) and start_hash.startswith("error:"):
            return _json.dumps({"error": start_hash[len("error:"):].strip()})
        if not start_hash or not end_hash:
            return _json.dumps({
                "error": (
                    "mark_document needs one of: tool_call_id, "
                    "start_hash+end_hash, or start_turn+end_turn"
                )
            })

        # Compute an original_tokens estimate for the dashboard.
        try:
            covered = self._store.get_messages_by_hashes([start_hash, end_hash])
            original_tokens = sum(
                len(str(m.get("content") or "")) // 4 + 10 for m in covered
            )
        except Exception:
            original_tokens = None

        meta: Dict[str, Any] = {
            "title": title,
            "source": "manual",
            "original_tokens": original_tokens,
        }
        # Strip None entries for a cleaner meta blob.
        meta = {k: v for k, v in meta.items() if v is not None}

        pin_id = self._store.add_pin(
            start_hash, end_hash,
            reason=title or "document",
            kind=kind,
            meta=meta,
        )
        return _json.dumps({
            "pin_id": pin_id,
            "kind": kind,
            "start_hash": start_hash,
            "end_hash": end_hash,
            "title": title,
            "original_tokens": original_tokens,
        })

    def _resolve_doc_range(
        self, args: Dict[str, Any],
    ) -> "tuple[Optional[str], Optional[str]]":
        """Map caller-provided identifiers to (start_hash, end_hash)."""
        store = self._store
        assert store is not None

        # Path 1: tool_call_id → single-message document.
        tool_call_id = args.get("tool_call_id")
        if tool_call_id:
            row = store.get_tool_result(tool_call_id)
            if not row:
                return (f"error: tool_call_id {tool_call_id} not found", None)
            return (row["hash"], row["hash"])

        # Path 2: explicit hash range.
        sh = args.get("start_hash")
        eh = args.get("end_hash")
        if sh and eh:
            # Validate both exist in the archive.
            rows = store.get_messages_by_hashes([sh, eh])
            found = {r["hash"] for r in rows}
            missing = [h for h in (sh, eh) if h not in found]
            if missing:
                return (f"error: hashes not archived: {missing}", None)
            return (sh, eh)

        # Path 3: turn range.
        st = args.get("start_turn")
        et = args.get("end_turn")
        if st is not None and et is not None:
            try:
                st, et = int(st), int(et)
            except (TypeError, ValueError):
                return ("error: start_turn and end_turn must be integers", None)
            if et < st:
                return ("error: end_turn must be >= start_turn", None)
            with store._lock:
                rows = store._conn.execute(
                    "SELECT hash, turn, position FROM messages "
                    "WHERE session_id = ? AND turn BETWEEN ? AND ? "
                    "ORDER BY turn, position",
                    (store.session_id, st, et),
                ).fetchall()
            if not rows:
                return (f"error: no messages between turns {st}..{et}", None)
            return (rows[0]["hash"], rows[-1]["hash"])

        return (None, None)

    # -- system prompt block -------------------------------------------

    def system_prompt_block(self) -> str:
        # Kept minimal for v1 — memory-search tool descriptions will
        # provide the user-visible surface area. Avoid noisy injections.
        return ""

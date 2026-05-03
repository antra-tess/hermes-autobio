"""Structural validators for the autobio engine.

Pure functions that inspect a live message list or archive state and
return a list of ``Issue`` objects. No exceptions — the harness
collects findings and keeps running so a single scenario surfaces
every defect rather than stopping on the first.

Two validators:

- :func:`validate_message_list` — provider-legality of the live view
  (tool pair integrity, role ordering, hash uniqueness, content shape).
  Cheap; run after every step.
- :func:`validate_archive` — cross-table consistency of the store
  (pin ranges resolvable, summary sources resolvable, FTS rows have
  parents, PRAGMA integrity_check passes). Run on transitions and at
  scenario end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from plugins.autobio.ids import message_hash
    from plugins.autobio.store import Store
except Exception:  # pragma: no cover
    message_hash = None
    Store = None


@dataclass
class Issue:
    severity: str       # "error" | "warn"
    category: str       # e.g. "orphan_tool_pair", "archive_integrity"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        ctx = ""
        if self.context:
            pairs = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            ctx = f" [{pairs}]"
        return f"{self.severity.upper():5s} {self.category}: {self.message}{ctx}"


# ---------------------------------------------------------------------------
# Live message list
# ---------------------------------------------------------------------------

_LEGAL_ROLES = {"system", "user", "assistant", "tool"}


def validate_message_list(messages: List[Dict[str, Any]]) -> List[Issue]:
    """Structural sanity of the live list.

    The checks here match what a chat-completions provider enforces:
    roles, pairing of tool_calls to tool_results, no stray tool
    messages, no duplicate content hashes.
    """
    issues: List[Issue] = []

    if not isinstance(messages, list):
        return [Issue("error", "shape", f"messages is not a list: {type(messages)!r}")]

    # System message, if present, must be at index 0.
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            issues.append(Issue(
                "error", "shape",
                f"message at index {i} is not a dict",
                {"index": i, "type": type(m).__name__},
            ))
            continue
        role = m.get("role")
        if role not in _LEGAL_ROLES:
            issues.append(Issue(
                "error", "role",
                f"illegal role {role!r} at index {i}",
                {"index": i},
            ))
        if role == "system" and i != 0:
            issues.append(Issue(
                "error", "role",
                f"system message at non-zero index {i}",
                {"index": i},
            ))

    # Tool-pair integrity. Walk the list collecting expected tool_call_ids
    # from assistant messages, then verifying every subsequent tool
    # message has a matching in-flight id (and that assistant groups
    # eventually get all their ids satisfied before the next user turn).
    in_flight: Dict[str, int] = {}       # tool_call_id -> index that opened it
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            for tc in tool_calls:
                tc_id = _tool_call_id(tc)
                if not tc_id:
                    issues.append(Issue(
                        "error", "tool_call_shape",
                        f"assistant at index {i} has a tool_call with no id",
                        {"index": i},
                    ))
                    continue
                if tc_id in in_flight:
                    issues.append(Issue(
                        "error", "duplicate_tool_call_id",
                        f"tool_call_id {tc_id!r} reopened at index {i}",
                        {"index": i, "first_open": in_flight[tc_id]},
                    ))
                in_flight[tc_id] = i
        elif role == "tool":
            tc_id = m.get("tool_call_id")
            if not tc_id:
                issues.append(Issue(
                    "error", "orphan_tool_result",
                    f"tool message at index {i} has no tool_call_id",
                    {"index": i},
                ))
                continue
            if tc_id not in in_flight:
                issues.append(Issue(
                    "error", "orphan_tool_result",
                    f"tool result at index {i} has no matching tool_call (id={tc_id!r})",
                    {"index": i, "tool_call_id": tc_id},
                ))
            else:
                in_flight.pop(tc_id, None)
        elif role in ("user", "system"):
            # Crossing a user/system boundary with unsatisfied tool_calls
            # is a hard error — the provider will reject this list.
            if in_flight:
                issues.append(Issue(
                    "error", "unsatisfied_tool_calls",
                    f"entered {role} at index {i} with open tool_calls {sorted(in_flight)}",
                    {"index": i, "open_ids": sorted(in_flight)},
                ))
                in_flight = {}

    if in_flight:
        issues.append(Issue(
            "warn", "unsatisfied_tool_calls",
            f"list ends with unsatisfied tool_calls: {sorted(in_flight)}",
            {"open_ids": sorted(in_flight)},
        ))

    # Hash uniqueness. Two identical messages in the live list is
    # almost always a bug — it means something got double-folded or a
    # summary marker was emitted twice.
    if message_hash is not None:
        seen: Dict[str, int] = {}
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            try:
                h = message_hash(m)
            except Exception:
                continue
            if h in seen:
                issues.append(Issue(
                    "warn", "duplicate_hash",
                    f"message at {i} duplicates earlier index {seen[h]}",
                    {"index": i, "earlier": seen[h], "hash": h},
                ))
            else:
                seen[h] = i

    # Content legality for role=assistant with no tool_calls — must have
    # non-empty content; empty-content assistant is illegal for most
    # providers unless tool_calls is populated.
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        tool_calls = m.get("tool_calls") or []
        content = m.get("content")
        if not tool_calls and (content is None or content == ""):
            issues.append(Issue(
                "warn", "empty_assistant",
                f"assistant at index {i} has no content and no tool_calls",
                {"index": i},
            ))

    return issues


def _tool_call_id(tc: Any) -> Optional[str]:
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def validate_archive(store: "Store") -> List[Issue]:
    """Cross-table consistency of the SQLite archive."""
    issues: List[Issue] = []
    if store is None:
        return [Issue("error", "archive", "store is None")]

    conn = getattr(store, "_conn", None) or getattr(store, "conn", None)
    if conn is None:
        return [Issue("warn", "archive", "cannot reach store connection for deep check")]

    # PRAGMA integrity_check.
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        result = row[0] if row else ""
        if str(result).lower() != "ok":
            issues.append(Issue(
                "error", "archive_integrity",
                f"PRAGMA integrity_check = {result!r}",
            ))
    except Exception as e:
        issues.append(Issue("error", "archive_integrity", f"integrity_check failed: {e}"))

    # Every pin's start/end hash must resolve in the messages table.
    try:
        for pin in store.list_pins():
            start_h = pin.get("start_hash")
            end_h = pin.get("end_hash")
            for label, h in (("start_hash", start_h), ("end_hash", end_h)):
                if not h:
                    continue
                row = conn.execute(
                    "SELECT 1 FROM messages WHERE hash = ? LIMIT 1", (h,),
                ).fetchone()
                if row is None:
                    issues.append(Issue(
                        "error", "pin_orphan",
                        f"pin {pin.get('id')} has {label}={h[:8]}… not in messages table",
                        {"pin_id": pin.get("id"), "hash": h, "kind": pin.get("kind")},
                    ))
    except Exception as e:
        issues.append(Issue("warn", "pin_check", f"could not audit pins: {e}"))

    # Every summary's source_hashes must resolve — either to another
    # summary id (for L2/L3) or a message hash (for L1).
    try:
        import json as _json
        rows = conn.execute(
            "SELECT id, level, source_hashes, source_hash FROM summaries"
        ).fetchall()
        summary_ids = {r[0] for r in rows}
        for sid, level, sh_json, src_hash in rows:
            try:
                src_list = _json.loads(sh_json) if sh_json else []
            except Exception:
                issues.append(Issue(
                    "error", "summary_source",
                    f"summary {sid} has malformed source_hashes JSON",
                    {"id": sid},
                ))
                continue
            for src in src_list:
                if not src:
                    continue
                if level == 1:
                    row = conn.execute(
                        "SELECT 1 FROM messages WHERE hash = ? LIMIT 1", (src,),
                    ).fetchone()
                    if row is None:
                        issues.append(Issue(
                            "error", "summary_source",
                            f"L1 {sid} references missing message {src[:8]}…",
                            {"summary_id": sid, "hash": src},
                        ))
                else:
                    if src not in summary_ids:
                        issues.append(Issue(
                            "error", "summary_source",
                            f"L{level} {sid} references missing child summary {src!r}",
                            {"summary_id": sid, "missing": src},
                        ))
    except Exception as e:
        issues.append(Issue("warn", "summary_check", f"could not audit summaries: {e}"))

    # FTS indexes should not have rows without a corresponding content row.
    # We check messages_fts ↔ messages and summaries_fts ↔ summaries.
    for fts_table, parent_table, parent_key, parent_col in (
        ("messages_fts", "messages", "rowid", "rowid"),
        ("summaries_fts", "summaries", "rowid", "rowid"),
    ):
        try:
            orphan = conn.execute(
                f"SELECT COUNT(*) FROM {fts_table} "
                f"WHERE {parent_key} NOT IN (SELECT {parent_col} FROM {parent_table})"
            ).fetchone()
            if orphan and orphan[0]:
                issues.append(Issue(
                    "warn", "fts_orphan",
                    f"{fts_table} has {orphan[0]} rows without a parent in {parent_table}",
                    {"count": orphan[0]},
                ))
        except Exception:
            # Some FTS5 configurations don't expose rowid in the content
            # table the same way; not a hard failure.
            pass

    # source_hash is supposed to be effectively unique per level — two
    # summaries at the same level pointing to the same chunk means the
    # dedup cache missed and we double-wrote.
    try:
        rows = conn.execute(
            "SELECT level, source_hash, COUNT(*) FROM summaries "
            "WHERE source_hash IS NOT NULL AND source_hash != '' "
            "GROUP BY level, source_hash HAVING COUNT(*) > 1"
        ).fetchall()
        for level, sh, count in rows:
            issues.append(Issue(
                "error", "duplicate_source_hash",
                f"level {level} has {count} summaries sharing source_hash {sh[:12]}…",
                {"level": level, "count": count, "source_hash": sh},
            ))
    except Exception:
        pass

    return issues


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def run_all(messages: List[Dict[str, Any]], store: "Store") -> List[Issue]:
    """Run both validators; convenience for harness."""
    return validate_message_list(messages) + validate_archive(store)

"""Archive-first tool result pruning.

Without pruning, a few hours of active tool use can flood the context
window with hundreds of kilobytes of terminal output and file reads.
This module is the primary mechanism that keeps the live message list
usable.

Three passes, matching the design spec:

  Pass 1 — Dedup identical tool results across the list. Older copies
           are replaced with a short back-reference; the newest full
           copy survives. Identity is content-hash.
  Pass 2 — Summarize tool results outside their retention window into
           informative 1-liners derived from tool name, args, and
           result metadata. No LLM call.
  Pass 3 — Truncate large tool_call arguments in assistant messages
           (e.g., ``write_file`` carrying 50KB of content) to
           ``argument_truncate_chars``.

Archive-first: every tool-role message that will be touched by a pass
is upserted into the ``messages`` table in full *before* the live list
is mutated, so the original is always recoverable via ``tool_call_id``.

Per-tool retention windows (from ``autobio.tool_pruning.per_tool``) are
applied per-tool-type by walking backward and accumulating tokens *of
that tool type*. Terminal can hold 50k of recent output even if reads
and searches have pushed everything else back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ids import message_hash
from .store import Store

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_DEDUP_MIN_CHARS = 200
_DUPLICATE_PLACEHOLDER = "[Duplicate tool output — same content as a more recent call]"


@dataclass
class PruneStats:
    dedup_count: int = 0
    summarized_count: int = 0
    truncated_args_count: int = 0
    archived_count: int = 0
    # (index, error_str) pairs for any tool-role message that failed
    # to archive in Step 0. Those positions are excluded from pruning
    # — losing both archive AND live copy is data loss we refuse.
    archive_failures: List["tuple[int, str]"] = field(default_factory=list)
    per_tool: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def bump_tool(self, tool: str, field_: str) -> None:
        slot = self.per_tool.setdefault(tool, {"full": 0, "summarized": 0})
        slot[field_] = slot.get(field_, 0) + 1


def prune_tool_results(
    messages: List[Dict[str, Any]],
    store: Optional[Store],
    cfg: Dict[str, Any],
    *,
    current_turn: int = 0,
) -> Tuple[List[Dict[str, Any]], PruneStats]:
    """Apply 3-pass archive-first pruning.

    ``cfg`` is the ``tool_pruning`` sub-dict from ``load_autobio_config()``.
    ``store`` may be ``None`` during tests — archival is skipped in that
    case but the passes still run.
    """
    stats = PruneStats()
    if not messages:
        return messages, stats

    # Build tool_call_id -> (tool_name, tool_args) from assistant messages.
    # Used by the summarizer to produce "[read_file] path=... (N chars)" lines.
    id_to_tool = _build_tool_call_index(messages)

    # Step 0: archive every tool-role message in full, keyed by content
    # hash. Idempotent via INSERT OR IGNORE so repeated compress() calls
    # don't write duplicates. Done BEFORE any pruning touches the list.
    #
    # Critical: if archival fails for a message, we must NOT prune that
    # message in subsequent passes — the principle is "archive-first."
    # Pruning a message that wasn't archived is irrecoverable data loss.
    # Archive failures are logged at error level (not debug) and the
    # offending positions are excluded from prune candidacy below.
    archived_ok: set[int] = set()
    archive_failures: List["tuple[int, str]"] = []
    if store is not None:
        from .tool_naming import build_tool_args_index, enrich_tool_name
        args_index = build_tool_args_index(messages)
        # Snapshot next-archive position once; only used when we need to
        # archive a not-yet-archived tool result. Reconcile and
        # _archive_live should already cover everything in normal
        # flow, but this defensive pass catches the edge cases.
        with store._lock:
            row = store._conn.execute(
                "SELECT COALESCE(MAX(position), -1) FROM messages "
                "WHERE session_id = ?",
                (store.session_id,),
            ).fetchone()
        next_pos = (row[0] if row else -1) + 1
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            try:
                content = msg.get("content", "")
                content_str = content if isinstance(content, str) else json.dumps(content)
                # Enrich tool_name with file-ext suffix for Read-family
                # tools BEFORE hashing so the hash and the stored row
                # agree, and so the picker's per-tool weight lookup can
                # tell ``Read:md`` (document — full weight) apart from
                # ``Read:py`` (source code — discounted).
                enriched = enrich_tool_name(msg, args_index)
                # Defensive default: if the args index didn't carry a
                # name (corrupt transcript, missing tool_call), at
                # least set ``unknown`` so downstream queries don't
                # confuse missing-name with empty content.
                if not enriched.get("name"):
                    enriched = dict(enriched)
                    enriched["name"] = id_to_tool.get(
                        msg.get("tool_call_id", ""), ("unknown", ""),
                    )[0] or "unknown"
                h = message_hash(enriched)
                # Already archived (by reconcile / _archive_live)? Skip
                # the upsert. With compound PK ``(session_id, hash,
                # position)``, passing the live-view index ``i`` here
                # would create a phantom row at a position the picker
                # doesn't expect, breaking source_hash continuity.
                if store.get_message(h) is None:
                    store.upsert_message(
                        h, enriched,
                        turn=current_turn, position=next_pos,
                        tokens=len(content_str) // _CHARS_PER_TOKEN,
                    )
                    next_pos += 1
                archived_ok.add(i)
                stats.archived_count += 1
            except Exception as e:
                logger.error(
                    "autobio tool_pruning: archive FAILED at index %d (%s) — "
                    "message will be EXCLUDED from pruning to avoid data loss",
                    i, e,
                )
                archive_failures.append((i, str(e)))
    else:
        # No store at all (test mode) — every message is "archive-not-required"
        # and pruning is allowed. Recording all positions makes the protection
        # check below a no-op when store is None.
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                archived_ok.add(i)

    if archive_failures:
        stats.archive_failures = archive_failures

    # Work on a shallow copy so the caller's list is not mutated.
    result = [m.copy() for m in messages]

    # Pass 1: deduplicate identical tool results — newest wins.
    result, n_dedup = _pass_dedup(result, archived_ok)
    stats.dedup_count = n_dedup

    # Pass 2: summarize results outside per-tool retention windows.
    result, n_summarized, per_tool_stats = _pass_summarize(
        result, cfg, id_to_tool, archived_ok=archived_ok,
    )
    stats.summarized_count = n_summarized
    for tool, counts in per_tool_stats.items():
        stats.per_tool.setdefault(tool, {"full": 0, "summarized": 0})
        for k, v in counts.items():
            stats.per_tool[tool][k] = stats.per_tool[tool].get(k, 0) + v

    # Pass 3: truncate oversized tool_call arguments in assistant messages
    # outside the retention window for their respective tools.
    result, n_truncated = _pass_truncate_args(result, cfg, id_to_tool)
    stats.truncated_args_count = n_truncated

    return result, stats


# ---------------------------------------------------------------------------
# Internal passes
# ---------------------------------------------------------------------------

def _build_tool_call_index(messages: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """Return ``{tool_call_id: (tool_name, arguments_json)}``."""
    index: Dict[str, Tuple[str, str]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                cid = tc.get("id", "")
                fn = tc.get("function") or {}
                index[cid] = (fn.get("name", "unknown"), fn.get("arguments", "") or "")
            else:
                cid = getattr(tc, "id", "") or ""
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "unknown") if fn else "unknown"
                args = getattr(fn, "arguments", "") if fn else ""
                index[cid] = (name, args or "")
    return index


def _pass_dedup(
    messages: List[Dict[str, Any]],
    archived_ok: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Replace older identical tool results with a placeholder.

    Walks backward: the first (newest) copy wins; subsequent copies get
    a back-reference. Only applied to results >= ``_DEDUP_MIN_CHARS`` to
    avoid mangling short stdout.

    ``archived_ok`` (when supplied) gates which indices are eligible
    for pruning — messages whose archive write failed in Step 0 stay
    in the live view unchanged so we don't lose the only copy.
    """
    seen: Dict[str, int] = {}
    pruned = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) < _DEDUP_MIN_CHARS:
            continue
        if content.startswith("[Duplicate tool output"):
            continue
        h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        if h in seen:
            if archived_ok is not None and i not in archived_ok:
                # Archive failed for this index — leave content intact.
                continue
            messages[i] = {**msg, "content": _DUPLICATE_PLACEHOLDER}
            pruned += 1
        else:
            seen[h] = i
    return messages, pruned


def _pass_summarize(
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    id_to_tool: Dict[str, Tuple[str, str]],
    archived_ok: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Dict[str, int]]]:
    """Replace old tool results with informative 1-liners.

    Retention is computed per-tool-type: walking backward, we accumulate
    content tokens separately for each tool. Once a tool type's
    accumulated tokens exceed its ``full_retention_tokens``, further
    messages of that tool type get summarized.

    Result messages that already carry the duplicate placeholder or a
    previously-generated summary are left alone.
    """
    default = (cfg.get("default") or {})
    per_tool_cfg = (cfg.get("per_tool") or {})
    summary_min_chars = int(default.get("summary_min_chars", 200))

    # Walk from newest to oldest, accumulating per-tool token counts.
    accumulated: Dict[str, int] = {}
    decisions: List[Tuple[int, str, bool]] = []  # (idx, tool_name, summarize?)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith(_DUPLICATE_PLACEHOLDER) or _looks_like_summary(content):
            continue
        tool_name, _ = id_to_tool.get(msg.get("tool_call_id", ""), ("unknown", ""))
        budget = int(
            per_tool_cfg.get(tool_name, {}).get(
                "full_retention_tokens",
                default.get("full_retention_tokens", 30_000),
            )
        )
        msg_tokens = len(content) // _CHARS_PER_TOKEN + 10
        already = accumulated.get(tool_name, 0)
        # If including this message keeps us under budget, it stays in full.
        if already + msg_tokens <= budget:
            accumulated[tool_name] = already + msg_tokens
            decisions.append((i, tool_name, False))
        else:
            decisions.append((i, tool_name, True))

    pruned = 0
    per_tool_stats: Dict[str, Dict[str, int]] = {}
    for i, tool_name, should_summarize in decisions:
        slot = per_tool_stats.setdefault(tool_name, {"full": 0, "summarized": 0})
        msg = messages[i]
        content = msg.get("content", "") or ""
        if not should_summarize:
            slot["full"] = slot.get("full", 0) + 1
            continue
        if len(content) < summary_min_chars:
            slot["full"] = slot.get("full", 0) + 1
            continue
        # Archive-first guard: never replace a message whose archive
        # upsert failed. We don't have a second copy to fall back on.
        if archived_ok is not None and i not in archived_ok:
            slot["full"] = slot.get("full", 0) + 1
            logger.error(
                "autobio: skipping summarize at idx %d — archive failed for this message",
                i,
            )
            continue
        call_id = msg.get("tool_call_id", "")
        _, tool_args = id_to_tool.get(call_id, (tool_name, ""))
        summary_line = summarize_tool_result(tool_name, tool_args, content)
        if not summary_line:
            # summarize_tool_result returned empty — refuse to fabricate.
            logger.error(
                "autobio: summarize_tool_result returned empty for tool=%s — "
                "leaving content intact rather than emit empty placeholder",
                tool_name,
            )
            slot["full"] = slot.get("full", 0) + 1
            continue
        messages[i] = {**msg, "content": summary_line}
        pruned += 1
        slot["summarized"] = slot.get("summarized", 0) + 1

    return messages, pruned, per_tool_stats


def _pass_truncate_args(
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    id_to_tool: Dict[str, Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Truncate oversized assistant tool_call arguments.

    Applied unconditionally when ``len(args) > argument_truncate_chars``
    because the bulk content is often in the args (``write_file`` with
    50KB of body) not the result. The most recent tool call of each
    tool type is exempted — the assistant may still be reasoning about
    what it just sent.

    Only dict-form tool_calls are touched. SimpleNamespace forms are
    provider-side response objects re-serialized by the request
    pipeline; leave them alone.
    """
    default = (cfg.get("default") or {})
    per_tool_cfg = (cfg.get("per_tool") or {})

    # Exempt the most recent call_id per tool_name so the assistant can
    # still see what it just sent.
    last_call_per_tool: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            cid = tc.get("id", "")
            if name and cid:
                last_call_per_tool[name] = cid

    truncated = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        new_tcs = []
        modified = False
        for tc in msg["tool_calls"]:
            if not isinstance(tc, dict):
                new_tcs.append(tc)
                continue
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "") or ""
            cid = tc.get("id", "")
            limit = int(
                per_tool_cfg.get(name, {}).get(
                    "argument_truncate_chars",
                    default.get("argument_truncate_chars", 500),
                )
            )
            exempt = last_call_per_tool.get(name) == cid
            if not exempt and len(args) > limit:
                new_fn = {**fn, "arguments": args[: max(limit - 15, 1)] + "...[truncated]"}
                new_tcs.append({**tc, "function": new_fn})
                modified = True
                truncated += 1
            else:
                new_tcs.append(tc)
        if modified:
            messages[i] = {**msg, "tool_calls": new_tcs}
    return messages, truncated


# ---------------------------------------------------------------------------
# 1-liner summary generation (no LLM — format-based)
# ---------------------------------------------------------------------------

def summarize_tool_result(tool_name: str, tool_args: str, content: str) -> str:
    """Produce an informative 1-line placeholder.

    Uses per-tool templates where a tool_name is known, falls back to a
    generic ``[name] arg=... (N chars)`` format otherwise. Mirrors the
    built-in compressor's ``_summarize_tool_result`` but keeps the
    format narrow and deterministic — no LLM call, no newlines.
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    content = content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output ({content_len:,} chars)"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written = args.get("content", "") or ""
        written_lines = written.count("\n") + 1 if written else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] '{query}' ({content_len:,} chars)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "memory_search" or tool_name == "memory_grep":
        query = args.get("query") or args.get("pattern", "?")
        return f"[{tool_name}] '{query}' ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars)"

    if tool_name == "execute_code":
        code = args.get("code", "") or ""
        preview = code[:60].replace("\n", " ")
        if len(code) > 60:
            preview += "..."
        return f"[execute_code] `{preview}` ({line_count} lines)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    # Generic fallback — include the first couple of arg key=value pairs.
    bits = []
    for k, v in list(args.items())[:2]:
        bits.append(f"{k}={str(v)[:40]}")
    detail = (" " + " ".join(bits)) if bits else ""
    return f"[{tool_name}]{detail} ({content_len:,} chars)"


def _looks_like_summary(text: str) -> bool:
    """Heuristic: does this look like an already-generated 1-line summary?"""
    return bool(re.match(r"^\[[a-zA-Z0-9_]+\][^\n]{0,300}$", text))

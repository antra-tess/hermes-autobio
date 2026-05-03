"""Live view as a projection of archive state.

Per the spec's first principle: the archive is authoritative; the live
view is a projection. ``render_view`` is the single function that
turns archive state into the message list the model sees on a given
turn.

Pure: no LLM calls, no mutation. Same archive state always yields
the same rendered list (modulo opts).

Algorithm:

1. Walk archived messages in ``position`` order.
2. For each message, compute its **active marker** — the highest-level
   non-merged summary covering it (chain via
   ``messages.compressed_into`` → ``summaries.merged_into``).
3. Group consecutive messages by active marker. Within a group:
   - active marker is None → emit each message verbatim as a raw
     conversation message.
   - active marker is a summary → emit one tiny marker for that
     summary in place of all the messages it covers.
4. Sanitize tool pairs (existing helper).

Tiny markers: a marker is a few-token reference, not the full L1
text. CM can fetch the full text via ``memory_read`` / ``UNCOMPRESS``.
This keeps the live tail proportional to "uncovered messages + N
small markers" instead of "uncovered + accumulated full L1 bodies."
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .store import Store

logger = logging.getLogger(__name__)


# Render-time markers carry an explicit ``_autobio_marker = True``
# dict key. That's the primary signal: schema, not content heuristic.
# Earlier revisions identified markers by content prefix alone, which
# meant any path that built a marker had to remember the exact prefix
# string and any path that consumed one had to know the heuristic.
# The flag turns "is this a marker?" into a structural answer that
# every code path agrees on.
#
# The prefix string is still useful as belt-and-braces for two cases:
# the dict survives a round-trip that drops unknown keys (rare — most
# message storage in Hermes is plain Python dicts) and very old stored
# state from before the flag was introduced. ``is_marker_message``
# falls back to the prefix at warning level.
_MARKER_PREFIX = "[autobio:memory "
_MARKER_FLAG = "_autobio_marker"


def render_view(
    store: Store,
    *,
    protect_first_n: int = 1,
    do_not_render_as_marker: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Render the live view from current archive state.

    Args:
        store: open Store for the session.
        protect_first_n: number of leading messages to always render
            raw (default 1, the system message).
        do_not_render_as_marker: optional set of summary ids that
            should NOT be collapsed to a marker even if they cover
            a region. The render will instead descend to children.
            Used by ``UNCOMPRESS`` semantics — once unfolded, a
            summary stays unfolded until refolded.

    Returns:
        A list of OpenAI-format message dicts. Tool pairs are
        sanitized so any orphaned tool_call/result is filtered.
    """
    do_not_render_as_marker = set(do_not_render_as_marker or ())

    rows = _all_messages(store)
    if not rows:
        return []

    # For every summary in the session, precompute its merge chain so
    # we can resolve the active marker for each message in O(1).
    summary_chain = _summary_chain(store)

    # Walk messages, grouping by active_summary_id.
    out: List[Dict[str, Any]] = []
    current_marker: Optional[str] = None
    pending_marker_summary: Optional[Dict[str, Any]] = None

    for idx, m in enumerate(rows):
        if idx < protect_first_n:
            # Always render the head raw.
            if current_marker is not None and pending_marker_summary is not None:
                out.append(_make_marker(pending_marker_summary))
            current_marker = None
            pending_marker_summary = None
            out.append(_msg_to_openai(m))
            continue

        active = _active_marker_for(m, summary_chain, do_not_render_as_marker)
        if active is None:
            # Flush any pending marker.
            if current_marker is not None and pending_marker_summary is not None:
                out.append(_make_marker(pending_marker_summary))
            current_marker = None
            pending_marker_summary = None
            out.append(_msg_to_openai(m))
        else:
            if active != current_marker:
                # Emit the previous marker (if any) before starting a new run.
                if current_marker is not None and pending_marker_summary is not None:
                    out.append(_make_marker(pending_marker_summary))
                current_marker = active
                pending_marker_summary = summary_chain["by_id"].get(active)

    # Flush trailing marker.
    if current_marker is not None and pending_marker_summary is not None:
        out.append(_make_marker(pending_marker_summary))

    return _sanitize_tool_pairs(out)


def known_marker_hashes(store: Store) -> Set[str]:
    """Return the set of message hashes that would be produced if every
    active summary were rendered as a marker right now.

    The engine uses this to recognize stale markers in input messages
    (from the runner's prior live state) and skip them on reconcile.
    """
    from .ids import message_hash
    chain = _summary_chain(store)
    out: Set[str] = set()
    for sid, summary in chain["by_id"].items():
        marker = _make_marker(summary)
        try:
            out.add(message_hash(marker))
        except Exception:
            pass
    return out


def is_marker_message(msg: Dict[str, Any]) -> bool:
    """True iff ``msg`` is a render-time autobio marker.

    Primary check: the explicit ``_autobio_marker`` flag set by
    ``_make_marker`` and the compression-layer marker constructor. Any
    fresh marker from this codebase has the flag.

    Fallback: the ``[autobio:memory `` content prefix. Logged at
    warning level when it fires alone — the flag should always be
    present on a fresh marker, so a prefix-only match means the flag
    was dropped somewhere upstream and the caller should investigate.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get(_MARKER_FLAG) is True:
        return True

    content = msg.get("content")
    matched_prefix = False
    if isinstance(content, str):
        matched_prefix = content.startswith(_MARKER_PREFIX)
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                t = blk.get("text") or ""
                if t.startswith(_MARKER_PREFIX):
                    matched_prefix = True
                break
    if matched_prefix:
        logger.warning(
            "autobio: marker recognized by prefix only (no %s flag) — "
            "investigate upstream caller dropping the flag. content=%r",
            _MARKER_FLAG,
            (content[:80] if isinstance(content, str) else "<list>"),
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _all_messages(store: Store) -> List[Dict[str, Any]]:
    """All messages for this session, in chronological position order."""
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY position ASC, turn ASC",
            (store.session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _summary_chain(store: Store) -> Dict[str, Any]:
    """Build per-session summary metadata for fast active-marker lookup.

    Returns:
        {
          "by_id": { sid: row_dict, ... },
          "merged_into": { sid: parent_sid_or_None, ... },
        }
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT id, level, source_hashes, merged_into, content, tokens, "
            "compression_generations, tags "
            "FROM summaries WHERE session_id = ?",
            (store.session_id,),
        ).fetchall()
    by_id: Dict[str, Dict[str, Any]] = {}
    merged_into: Dict[str, Optional[str]] = {}
    for r in rows:
        d = dict(r)
        by_id[d["id"]] = d
        merged_into[d["id"]] = d.get("merged_into")
    return {"by_id": by_id, "merged_into": merged_into}


def _active_marker_for(
    msg_row: Dict[str, Any],
    chain: Dict[str, Any],
    do_not_render: Set[str],
) -> Optional[str]:
    """Walk the merge chain to find the highest active summary covering
    this message, skipping any in do_not_render. Returns None if the
    message isn't covered (still raw L0)."""
    sid = msg_row.get("compressed_into")
    if not sid or sid not in chain["by_id"]:
        return None
    # Walk upward through merged_into until we hit one whose merged_into
    # is None (active) or one in do_not_render (descend instead).
    current = sid
    last_ok: Optional[str] = None
    while current is not None:
        parent = chain["merged_into"].get(current)
        # Track the last summary that's NOT in do_not_render — that's
        # what we'd render. If everything up the chain is do_not_render,
        # we return None (meaning render raw / children).
        if current not in do_not_render:
            last_ok = current
        if parent is None:
            break
        current = parent
    if last_ok is None:
        # Every level is forced expanded; expose raw messages.
        return None
    # If last_ok itself is unfolded, expose raw.
    if last_ok in do_not_render:
        return None
    return last_ok


def _make_marker(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the tiny render-time marker for a summary.

    ~50 tokens. Carries the id, level, token cost, generation, and a
    short hint about how to fetch the full content. NOT the full L1
    body — that lives in the archive and is fetched on demand.

    The dict carries an explicit ``_autobio_marker`` flag so reconcile
    paths can recognize markers by schema rather than content prefix.
    """
    sid = summary["id"]
    level = int(summary.get("level", 1))
    tokens = int(summary.get("tokens", 0) or 0)
    gen = int(summary.get("compression_generations", 1) or 1)
    body = (
        f"{_MARKER_PREFIX}{sid} · L{level} · {tokens}t · gen {gen} · "
        f"call cm_manage_context to expand]"
    )
    return {"role": "user", "content": body, _MARKER_FLAG: True}


def _msg_to_openai(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an archive row into an OpenAI-format message dict.

    Tool calls are stored in canonical OpenAI shape (``{id, type,
    function: {name, arguments}}``) — see ``_canonicalize_tool_calls``
    in ``store.py`` and the schema migration that backfills any
    pre-canonical rows. Reading is now a flat JSON-decode and
    pass-through; no shape translation happens here.

    A defensive re-inflation is kept for the legacy flat shape (no
    ``function`` key). It should never fire on a migrated store, but
    if it does we log at warning level so the operator notices that
    something wrote a flat-form row past the migration boundary.
    """
    out: Dict[str, Any] = {"role": row.get("role", "user")}
    content = row.get("content")
    if isinstance(content, str) and content.startswith("["):
        # Possibly JSON-encoded multimodal content — try parsing.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                content = parsed
        except Exception:
            pass
    out["content"] = content if content is not None else ""
    if row.get("tool_call_id"):
        out["tool_call_id"] = row["tool_call_id"]
    if row.get("tool_calls"):
        try:
            raw_tcs = json.loads(row["tool_calls"])
        except (TypeError, json.JSONDecodeError):
            raw_tcs = None
        if isinstance(raw_tcs, list):
            tcs: List[Dict[str, Any]] = []
            saw_legacy = False
            for tc in raw_tcs:
                if not isinstance(tc, dict):
                    continue
                if isinstance(tc.get("function"), dict):
                    tcs.append(tc)
                    continue
                # Legacy flat form sneaked past the migration. Inflate
                # so the rest of the pipeline still sees canonical
                # shape, but log so the gap can be investigated.
                saw_legacy = True
                tcs.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    },
                })
            if saw_legacy:
                logger.warning(
                    "autobio render: legacy flat-form tool_calls in archive "
                    "row %s — store migration may not have run on this DB",
                    row.get("hash"),
                )
            if tcs:
                out["tool_calls"] = tcs
    if row.get("tool_name") and row.get("role") == "tool":
        out["name"] = row["tool_name"]
    if row.get("reasoning_details"):
        try:
            out["reasoning_details"] = json.loads(row["reasoning_details"])
        except (TypeError, json.JSONDecodeError):
            pass
    return out


def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter orphan tool_call/result pairs after rendering.

    OpenAI's sequence rule: every assistant tool_call must have its
    matching tool_result appear before the next user/system message.
    A render that drops a chunk-of-messages can break this if the
    chunk boundary sat between a tool_call and its result.

    Strategy: walk left-to-right. For each assistant with tool_calls,
    look ahead until the next user/system; collect the tool_call_ids
    that have a matching tool_result in that lookahead window. Drop
    tool_calls whose match isn't in the window. Drop tool messages
    whose call doesn't precede them within the same window.
    """
    keep: List[Dict[str, Any]] = []
    n = len(messages)
    i = 0
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "assistant" and (m.get("tool_calls") or []):
            tcs = m["tool_calls"]
            # Lookahead: next user/system OR end-of-list.
            j = i + 1
            available_results: set = set()
            while j < n and messages[j].get("role") not in ("user", "system"):
                if messages[j].get("role") == "tool":
                    cid = messages[j].get("tool_call_id")
                    if cid:
                        available_results.add(cid)
                j += 1
            kept_tcs = [
                tc for tc in tcs
                if isinstance(tc, dict) and tc.get("id") in available_results
            ]
            if kept_tcs:
                copy = dict(m)
                copy["tool_calls"] = kept_tcs
                keep.append(copy)
                # Now emit the matching tool messages and any narrative
                # assistants in this window, skipping orphan tools.
                kept_ids = {tc.get("id") for tc in kept_tcs}
                for k in range(i + 1, j):
                    other = messages[k]
                    other_role = other.get("role")
                    if other_role == "tool":
                        if other.get("tool_call_id") in kept_ids:
                            keep.append(other)
                    else:
                        # Non-tool messages in the lookahead (rare —
                        # interleaved assistant narrations) pass through.
                        keep.append(other)
                i = j
                continue
            elif m.get("content"):
                copy = dict(m)
                copy.pop("tool_calls", None)
                keep.append(copy)
            # else: drop the assistant entirely
            # Skip the tool results that have no opener (orphans).
            for k in range(i + 1, j):
                other = messages[k]
                if other.get("role") != "tool":
                    keep.append(other)
            i = j
            continue
        if role == "tool":
            # Orphan tool result not preceded by a matching assistant
            # in the same window — drop.
            i += 1
            continue
        keep.append(m)
        i += 1
    return keep

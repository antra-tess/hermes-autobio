"""Live view as a projection of archive state.

Per the spec's first principle: the archive is authoritative; the live
view is a projection. ``render_view`` is the single function that
turns archive state into the message list the model sees on a given
turn.

Pure: no LLM calls, no mutation. Same archive state always yields
the same rendered list (modulo opts).

## Terminology

Two distinct concepts that earlier versions of the code conflated
under the word "marker":

- **marker** — the in-band priming text the SUMMARIZER prompt
  carries (``compression._MARKER`` = "System: You will soon form a
  new memory, get ready..."). Visible only during the LLM call that
  forms a memory; tells the summarizer which slice of the prompt to
  write a memory for. **Not** part of the live view at all.
- **recall pair** — the live-view representation of a folded
  region. A two-turn dialogue placed where the original messages
  would have been: ``[user "[CM] Recall memory L1-X (gen N)."]`` +
  ``[assistant "<first-person body>"]``. Same shape as the
  CM-asks/agent-recalls pattern used inside summarizer prompts.
  This module is concerned exclusively with recall pairs.

## Algorithm

1. Walk archived messages in ``position`` order.
2. For each message, look at ``compressed_into`` directly — the L1
   the message was folded into. We do not walk up to L2/L3:
   "fill lower orbitals first" / adaptive-resolution rule. The L1
   is the highest-fidelity available representation; the L2/L3
   structure exists only for budget-constrained scenarios that
   v1 doesn't yet handle.
3. Group consecutive messages by their L1. For each group emit ONE
   recall pair for that L1.
4. Uncovered messages (``compressed_into IS NULL``) and messages
   whose L1 has been explicitly unfolded pass through verbatim.
5. Sanitize tool pairs (existing helper).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .store import Store

logger = logging.getLogger(__name__)


# Recall-pair recognition. The header (CM-voice user turn) starts
# with this prefix; both halves of the pair carry the
# ``_autobio_recall`` flag. The flag is the primary signal:
# structural, not content-heuristic. The prefix is a belt-and-braces
# fallback for cases where a round-trip drops unknown dict keys.
_RECALL_HEADER_PREFIX = "[CM] Recall memory "
_RECALL_FLAG = "_autobio_recall"


def render_view(
    store: Store,
    *,
    protect_first_n: int = 1,
    unfolded_summaries: Optional[Set[str]] = None,
    # Backwards-compatible alias for callers still passing the old
    # parameter name. ``unfolded_summaries`` is the canonical name.
    do_not_render_as_marker: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Render the live view from current archive state.

    Args:
        store: open Store for the session.
        protect_first_n: number of leading messages to always render
            raw (default 1, the system message).
        unfolded_summaries: optional set of L1 IDs that should NOT be
            collapsed to a recall pair — instead the underlying raw
            messages are emitted in place. Used by ``UNCOMPRESS``
            semantics: once unfolded, an L1 stays unfolded until
            explicitly refolded.

    Returns:
        A list of OpenAI-format message dicts. Each folded region is
        represented by a 2-turn recall pair (CM-voice user header +
        assistant body); raw messages pass through verbatim. Tool
        pairs are sanitized so any orphaned tool_call/result is
        filtered.
    """
    if unfolded_summaries is None:
        unfolded_summaries = set(do_not_render_as_marker or ())
    else:
        unfolded_summaries = set(unfolded_summaries)
        if do_not_render_as_marker:
            unfolded_summaries |= set(do_not_render_as_marker)

    rows = _all_messages(store)
    if not rows:
        return []

    # Per-session summary metadata: id → row dict, for body lookup.
    by_id = _summary_by_id(store)

    out: List[Dict[str, Any]] = []
    current_l1: Optional[str] = None
    pending_l1_summary: Optional[Dict[str, Any]] = None

    def _flush_pending() -> None:
        nonlocal current_l1, pending_l1_summary
        if pending_l1_summary is not None:
            out.extend(_make_recall_pair(pending_l1_summary))
        current_l1 = None
        pending_l1_summary = None

    for idx, m in enumerate(rows):
        if idx < protect_first_n:
            _flush_pending()
            out.append(_msg_to_openai(m))
            continue

        l1_id = m.get("compressed_into")
        if (
            not l1_id
            or l1_id not in by_id
            or l1_id in unfolded_summaries
        ):
            _flush_pending()
            out.append(_msg_to_openai(m))
            continue

        # Message is covered by an L1. Group consecutive same-L1
        # messages so we emit one recall pair per group.
        if l1_id != current_l1:
            _flush_pending()
            current_l1 = l1_id
            pending_l1_summary = by_id.get(l1_id)
        # else: same L1 group continues — skip (recall pair pending).

    _flush_pending()
    return _sanitize_tool_pairs(out)


def known_recall_hashes(store: Store) -> Set[str]:
    """Hashes of every message that would be emitted as part of a
    recall pair if the live view were rendered right now.

    The engine uses this to recognize round-tripped recall pair
    messages in the runner's conversation_history and skip them on
    reconcile (so a recall pair isn't re-archived as if it were a new
    conversation turn).
    """
    from .ids import message_hash
    by_id = _summary_by_id(store)
    out: Set[str] = set()
    for sid, summary in by_id.items():
        if int(summary.get("level") or 0) != 1:
            # Recall pairs are emitted for L1 only; L2/L3 are
            # invisible in the live view (their constituent L1s are
            # surfaced directly per the balance rule).
            continue
        try:
            for piece in _make_recall_pair(summary):
                out.add(message_hash(piece))
        except Exception:
            pass
    return out


def is_recall_message(msg: Dict[str, Any]) -> bool:
    """True iff ``msg`` is one half of a render-time recall pair.

    Primary check: the explicit ``_autobio_recall`` flag set on both
    turns by ``_make_recall_pair``. Any fresh recall pair has the
    flag.

    Fallback: the ``[CM] Recall memory `` content prefix on the
    header turn. The body turn has no distinctive prefix; if the
    flag is missing on the body, we cannot reliably recognize it and
    the warning will fire once on the header to alert the operator
    that something upstream is stripping unknown dict keys.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get(_RECALL_FLAG) is True:
        return True

    # Prefix fallback (header only).
    content = msg.get("content")
    matched_prefix = False
    if isinstance(content, str):
        matched_prefix = content.startswith(_RECALL_HEADER_PREFIX)
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                t = blk.get("text") or ""
                if t.startswith(_RECALL_HEADER_PREFIX):
                    matched_prefix = True
                break
    if matched_prefix:
        logger.warning(
            "autobio: recall header recognized by prefix only "
            "(no %s flag) — investigate upstream caller dropping "
            "the flag. content=%r",
            _RECALL_FLAG,
            (content[:80] if isinstance(content, str) else "<list>"),
        )
        return True
    return False


# Legacy aliases — kept so external callers (HOOK consumers,
# experimental engines) that import the old names don't break
# while we migrate. Will be removed in a follow-up release. The
# warning mirrors the deprecation pattern used elsewhere.
def is_marker_message(msg: Dict[str, Any]) -> bool:
    """Deprecated alias for :func:`is_recall_message`."""
    return is_recall_message(msg)


def known_marker_hashes(store: Store) -> Set[str]:
    """Deprecated alias for :func:`known_recall_hashes`."""
    return known_recall_hashes(store)


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


def _summary_by_id(store: Store) -> Dict[str, Dict[str, Any]]:
    """Per-session summary metadata, keyed by id. Used to look up the
    body of an L1 when emitting a recall pair.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT id, level, source_hashes, merged_into, content, tokens, "
            "compression_generations, tags "
            "FROM summaries WHERE session_id = ?",
            (store.session_id,),
        ).fetchall()
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = dict(r)
        by_id[d["id"]] = d
    return by_id


def _make_recall_pair(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the two-turn dialogue that represents a folded region in
    the live view.

    Output: ``[header_turn, body_turn]``.

    - **Header turn** (CM-voice user message): tells the agent which
      memory is being surfaced. Carries the L1's id and generation
      count so the LLM can refer to them. Text starts with the
      ``_RECALL_HEADER_PREFIX`` sentinel for prefix-fallback
      recognition.
    - **Body turn** (assistant message): the agent's first-person
      memory content, exactly as the summarizer wrote it.

    Both turns carry the ``_autobio_recall`` flag for structural
    recognition on round-trip.

    Role choice rationale:

    - Header is in user role: it is NOT produced by the LLM; it is a
      synthetic CM-voice prompt asking the agent to recall. Putting
      it under ``assistant`` would falsely attribute system metadata
      to the agent.
    - Body is in assistant role: it IS the agent's autobiographical
      recall (first-person, written by the summarizer). Putting it
      under ``user`` would tell the model "the user said this
      happened" rather than "I recall this happening" — a
      meaningful KV-level distortion.

    Anthropic dropped the strict role-alternation rule, so
    consecutive same-role turns concatenate naturally and we don't
    need any of the workarounds the older marker design carried.
    """
    sid = summary["id"]
    gen = int(summary.get("compression_generations", 1) or 1)
    body = summary.get("content") or ""
    header = {
        "role": "user",
        "content": f"{_RECALL_HEADER_PREFIX}{sid} (gen {gen}).",
        _RECALL_FLAG: True,
    }
    body_turn = {
        "role": "assistant",
        "content": body,
        _RECALL_FLAG: True,
    }
    return [header, body_turn]


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
            # Lookahead: next user/system, OR the next
            # assistant-with-tool_calls (which starts its own window)
            # OR end-of-list. Without the assistant-with-tool_calls
            # break, the first assistant's lookahead would swallow
            # the SECOND assistant's tool results — passing the
            # ``in available_results`` check globally but failing the
            # ``in kept_ids`` per-assistant check below, dropping the
            # second tool_result entirely. (Bug observed in production
            # when the agent ran two memory_search calls back-to-back.)
            j = i + 1
            available_results: set = set()
            while j < n and messages[j].get("role") not in ("user", "system"):
                msg_j = messages[j]
                if (msg_j.get("role") == "assistant"
                        and msg_j.get("tool_calls")):
                    break
                if msg_j.get("role") == "tool":
                    cid = msg_j.get("tool_call_id")
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

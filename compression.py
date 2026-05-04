"""Hierarchical summarization — L0→L1 and L1→L2→L3 merging.

Two primitives, cleanly separated:

- ``summarize_chunk`` is an **archive** operation. It produces an L1
  for a slice of messages and writes it to the store. It is idempotent
  via source_hash lookup — if the chunk has already been summarized,
  the existing L1 is returned without calling the LLM. Safe to run
  eagerly or in the background, regardless of whether the chunk is
  pinned or not: summarizing never mutates the live view.

- ``fold_chunk`` is a **display** operation. It mutates the live
  message list, replacing a chunk with a summary marker. This is the
  decision gated by pin rules and token pressure. Folding a range
  that has never been summarized is an error; summarize first.

- ``unfold`` is the inverse of fold. Given a summary ID, it finds the
  marker in the live list and expands back to source messages from
  the archive.

- ``compress_step`` is the orchestrator that the engine's compress()
  calls: pick chunk bounds, summarize, fold. It preserves the
  previous public API.

The summarization prompt is in strict as-of-the-chunk order — the
chunk is the newest thing in the summarizer's context, nothing after
it is visible. See ``_build_kv_preserving_prompt`` for details.

The marker ("System: You will soon form a new memory, get ready") is
an in-band signal, not a system prompt. It primes attention without
disrupting the conversational KV state. Subsequent check-ins and
compressions refer back to the same marker form, so the model treats
memory formation as a recurring narrated event rather than a fresh
instruction each time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.model_metadata import estimate_messages_tokens_rough

from .ids import combined_hash, message_hash
from .store import Store

logger = logging.getLogger(__name__)


_CHARS_PER_TOKEN = 4
_MARKER = (
    "System: You will soon form a new memory, get ready. "
    "The messages that follow are the slice of recent experience you are "
    "about to compress. After them, write the memory in your own voice."
)
_INSTRUCTION = (
    "Write the memory of events since the most recent memory system "
    "notification. Speak in the first person from your own perspective. "
    "Preserve concrete details — file paths, exact values, decisions, "
    "unresolved questions, the user's active asks. Target ~{target_tokens} "
    "tokens. Output only the memory body — no preamble, no section headers "
    "unless they help preservation, no meta-commentary about summarizing."
)

_MERGE_INSTRUCTION = (
    "You are consolidating {n} earlier memories into a single higher-level "
    "memory. The memories below are in chronological order. Write a new "
    "memory that preserves the through-line: what happened, what was "
    "decided, what remains open, what concrete details future you will "
    "want to reach for. Speak in the first person. Target ~{target_tokens} "
    "tokens. Output only the memory body."
)

_DOC_CHUNK_INSTRUCTION = (
    "You just read a slice of document \"{title}\" ({style}, approximately "
    "{original_tokens} tokens total in the full document). Earlier slices "
    "of this same document are in your memory above as L1s — they are in "
    "chronological order within the document. What comes after this slice "
    "you have NOT yet read; treat yourself as reading the document in "
    "order, not yet knowing what lies ahead.\n\n"
    "Write the memory of what this slice contained. Preserve concrete "
    "detail aggressively:\n"
    "- Timestamps, dates, IDs, numeric values, quoted phrases\n"
    "- Key events, decisions, contradictions with earlier slices\n"
    "- Structural markers (section titles, entry boundaries) that help "
    "you locate this slice within the whole later\n\n"
    "Speak in the first person. Target ~{target_tokens} tokens. Output "
    "only the memory body — no preamble, no meta-commentary about "
    "summarizing."
)


class SummarizationFailed(Exception):
    """Raised when the summarizer returns no usable content.

    Callers must NOT fabricate a placeholder summary to keep the pipeline
    moving — the archive-is-authoritative invariant forbids manufactured
    memories. Instead, ``engine.compress()`` catches this, sets a halt
    flag that blocks further compression attempts, and injects a visible
    assistant message so the failure is surfaced to the main instance
    and the human, not silently absorbed.
    """

    def __init__(self, chunk_source_hash: str, *, level: int = 1,
                 reason: str = "empty summarizer output") -> None:
        self.chunk_source_hash = chunk_source_hash
        self.level = level
        self.reason = reason
        super().__init__(
            f"summarization failed at L{level} for chunk {chunk_source_hash[:12]}: {reason}"
        )


@dataclass
class ModelInfo:
    """Runtime info for the summarization LLM call."""

    model: str = ""
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    api_mode: str = ""
    summary_model_override: str = ""


@dataclass
class CompressResult:
    new_messages: List[Dict[str, Any]]
    summary_id: Optional[str]
    chunk_tokens: int
    summary_tokens: int
    saved_tokens: int


# ---------------------------------------------------------------------------
# L0 → L1
# ---------------------------------------------------------------------------

def summarize_chunk(
    messages: List[Dict[str, Any]],
    chunk_start: int,
    chunk_end: int,
    store: Store,
    cfg: Dict[str, Any],
    model_info: ModelInfo,
    *,
    first_turn: Optional[int] = None,
    last_turn: Optional[int] = None,
    doc_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Produce an L1 summary of a chunk and persist to archive.

    This is the **archive write** half of compression. It does not
    modify ``messages`` — the live view is untouched. Safe to run
    eagerly in the background, including for pinned content.

    Idempotent: if an L1 already exists for this chunk's source_hash,
    returns that L1's ID without calling the LLM.

    ``doc_context`` — optional. When the chunk comes from the top of
    a document pin, pass the pin's metadata here (``pin_id``, ``title``,
    ``style``, ``original_tokens``). The summarizer receives a doc-
    tailored instruction emphasising chronological framing and detail
    preservation, and the resulting L1 is tagged for grouping in
    dashboards.

    Returns the summary ID, or None if summarization failed.
    """
    if chunk_start >= chunk_end:
        return None
    chunk = messages[chunk_start:chunk_end]
    chunk_hashes = [message_hash(m) for m in chunk]
    chunk_source_hash = combined_hash(chunk_hashes)

    # Defensive archive: any caller (main compress, speculative worker,
    # direct test invocation) must end up with the chunk's messages in
    # the archive — otherwise the L1's source_hashes point to rows
    # that don't exist. upsert_message is INSERT OR IGNORE, so if the
    # engine already archived these (via compress() or on_turn_end),
    # these calls are no-ops. We deliberately pass turn=None here —
    # correct turn numbers are set by the archival path that knew the
    # turn counter at the time. Overwriting with a guess would corrupt
    # the derived turn-range lookup.
    upsert_failures: List[int] = []
    for offset, m in enumerate(chunk):
        try:
            store.upsert_message(
                chunk_hashes[offset], m,
                turn=None,
                position=chunk_start + offset,
            )
        except Exception as e:
            logger.error(
                "summarize_chunk: upsert FAILED at offset %d: %s", offset, e,
            )
            upsert_failures.append(offset)

    # Verify every chunk hash actually resolves in the archive — this
    # catches both upsert failures above AND any prior archival gap
    # (e.g., engine.on_turn_end didn't run, or a test bypassed it).
    # Writing a summary whose source_hashes don't resolve violates
    # archive-is-authoritative: a future expand_summary would silently
    # return an incomplete reconstruction. Halt instead.
    missing = _missing_hashes(store, chunk_hashes)
    if missing:
        logger.error(
            "summarize_chunk: %d/%d chunk message hashes are NOT in the "
            "archive (sample: %s) — refusing to write a summary that "
            "references missing source content. Halting.",
            len(missing), len(chunk_hashes), [h[:12] for h in missing[:3]],
        )
        raise SummarizationFailed(
            chunk_source_hash, level=1,
            reason=f"{len(missing)} of {len(chunk_hashes)} chunk messages "
                   "not in archive (upsert failures or prior archival gap)",
        )

    # Archive dedup: the chunk already has a committed L1, return it.
    existing = store.find_summary_by_source_hash(chunk_source_hash, level=1)
    if existing is not None:
        logger.debug(
            "autobio summarize_chunk: reusing existing %s for chunk source_hash=%s",
            existing["id"], chunk_source_hash,
        )
        return existing["id"]

    summary_target = int(cfg.get("compression", {}).get("summary_target_tokens", 2_500))
    chunk_tokens = estimate_messages_tokens_rough(chunk)
    logger.info(
        "autobio summarize: chunk size=%d tokens~%d (target ~%d)%s",
        chunk_end - chunk_start, chunk_tokens, summary_target,
        f" [doc:{doc_context.get('title') or doc_context.get('pin_id')}]"
        if doc_context else "",
    )

    prompt_messages = _build_kv_preserving_prompt(
        messages=messages,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        store=store,
        target_tokens=summary_target,
        doc_context=doc_context,
    )
    # Reasoning models (e.g. K2.6, o1) consume hidden completion tokens
    # before emitting content; the 1.4× budget is too tight for them.
    # Allow cfg override via compression.max_tokens_override.
    max_tokens_override = cfg.get("compression", {}).get("max_tokens_override")
    max_tokens = (int(max_tokens_override) if max_tokens_override
                  else int(summary_target * 1.4))
    # Reasoning models occasionally return empty content even on
    # successful API calls (truncated reasoning, model wobble). A
    # single empty result is rarely meaningful; sustained empty
    # results across retries means the auxiliary is genuinely broken.
    # Retry quietly N times before raising.
    retries = int(cfg.get("compression", {}).get("summarize_retries", 3))
    summary_text = ""
    last_call_id: Optional[int] = None
    for attempt in range(1, retries + 1):
        summary_text, last_call_id = _call_summarizer(
            prompt_messages, model_info, max_tokens=max_tokens,
            store=store, purpose="summarize",
            chunk_source_hash=chunk_source_hash,
        )
        if summary_text:
            if attempt > 1:
                logger.info(
                    "autobio summarize: succeeded on attempt %d/%d for chunk %s",
                    attempt, retries, chunk_source_hash,
                )
            break
        logger.warning(
            "autobio summarize: empty output on attempt %d/%d for chunk %s "
            "(model=%s, max_tokens=%d)",
            attempt, retries, chunk_source_hash,
            model_info.model or "?", max_tokens,
        )
    if not summary_text:
        # Halt rather than fabricate. See SummarizationFailed docstring.
        logger.error(
            "autobio summarize: empty after %d retries for chunk %s "
            "(model=%s) — halting compression",
            retries, chunk_source_hash, model_info.model or "?",
        )
        raise SummarizationFailed(
            chunk_source_hash, level=1,
            reason=f"empty summarizer output after {retries} retries",
        )

    summary_tokens = max(1, len(summary_text) // _CHARS_PER_TOKEN)
    tags: Optional[List[str]] = None
    if doc_context is not None:
        tags = [f"doc:{doc_context.get('pin_id', '?')}"]
        if doc_context.get("title"):
            tags.append(f"title:{doc_context['title']}")

    summary_id = store.insert_summary(
        level=1,
        source_hashes=chunk_hashes,
        source_hash=chunk_source_hash,
        content=summary_text,
        tokens=summary_tokens,
        compression_generations=1,
        tags=tags,
    )
    if last_call_id is not None and summary_id is not None:
        try:
            store.attach_summary_to_llm_call(last_call_id, summary_id)
        except Exception as e:
            logger.debug("attach_summary_to_llm_call failed: %s", e)
    logger.info(
        "autobio L1 written: %s (%d tokens) for chunk of %d tokens",
        summary_id, summary_tokens, chunk_tokens,
    )
    return summary_id


def fold_chunk(
    messages: List[Dict[str, Any]],
    chunk_start: int,
    chunk_end: int,
    summary_id: str,
    store: Store,
) -> List[Dict[str, Any]]:
    """Substitute a chunk in the live view with its L1 summary marker.

    This is the **display mutation** half. Does not call the LLM. The
    summary must already exist in the archive; callers should run
    summarize_chunk first (or verify via find_summary_by_source_hash).

    Order is critical: ``mark_compressed`` must succeed BEFORE the
    live list is mutated. If we folded first and then failed to mark,
    the live view would lose the messages with no uncompression path
    to recover them. ``mark_compressed`` raises on bad ``summary_id``,
    which we let propagate so the caller doesn't return half-folded
    state.
    """
    summary = store.get_summary(summary_id)
    if summary is None:
        raise ValueError(f"fold_chunk: summary {summary_id} not found in archive")
    summary_text = summary["content"]

    # Step 1 (DB): mark source messages compressed_into. If this fails
    # (summary missing/merged/db locked), abort BEFORE mutating the
    # live list.
    chunk_hashes = [message_hash(m) for m in messages[chunk_start:chunk_end]]
    store.mark_compressed(chunk_hashes, summary_id)

    # Step 2 (in-memory): swap the live view.
    marker = _make_summary_marker(summary_id, summary_text, level=int(summary["level"]))
    new_messages = messages[:chunk_start] + [marker] + messages[chunk_end:]
    return _sanitize_tool_pairs(new_messages)


def unfold(
    messages: List[Dict[str, Any]],
    summary_id: str,
    store: Store,
) -> List[Dict[str, Any]]:
    """Inverse of fold: replace a summary marker in the live view with
    its source messages restored from archive.

    Locates the marker for ``summary_id`` in ``messages`` and swaps it
    out for the original content. For L1s the sources are raw messages;
    for L2/L3 they are child summary markers (rendered as user-role
    pseudo-messages so the sequence stays well-formed).

    Returns the list unchanged if no marker is found. Re-raises
    ``UnfoldFailed`` so the caller can surface incomplete archives
    rather than silently restoring partial content.
    """
    from .search import UnfoldFailed, expand_summary

    marker_idx = _find_marker_index(messages, summary_id)
    if marker_idx is None:
        logger.debug("autobio unfold: no marker for %s in live view", summary_id)
        return messages

    try:
        expanded = expand_summary(store, summary_id)
    except UnfoldFailed:
        # Propagate. The caller (CM, search-tool) decides how to surface.
        raise
    if not expanded:
        return messages

    new_messages = messages[:marker_idx] + expanded + messages[marker_idx + 1:]
    return _sanitize_tool_pairs(new_messages)


def compress_step(
    messages: List[Dict[str, Any]],
    store: Store,
    cfg: Dict[str, Any],
    model_info: ModelInfo,
    *,
    protect_first_n: int = 3,
    protect_last_tokens: Optional[int] = None,
    is_eligible: Optional[Callable[[Dict[str, Any]], bool]] = None,
    # first_turn/last_turn are accepted for backcompat with older
    # callers but no longer stored on the summary. Turn ranges are
    # derived on demand via store.get_summary_turn_range.
    first_turn: Optional[int] = None,     # noqa: ARG001 — deprecated
    last_turn: Optional[int] = None,      # noqa: ARG001 — deprecated
) -> CompressResult:
    """Pick a chunk, summarize it, fold it. Engine's compress() entry.

    Consults the store's pins table to find protected ranges in the
    current live view. Plain pins and ineligible docs are skipped; the
    picker returns the oldest foldable region.

    ``is_eligible(range) -> bool`` decides whether a document is
    foldable on this step. Default: never (preserving docs). Pressure-
    based policies are layered on top by the engine.
    """
    compression_cfg = cfg.get("compression", {})
    chunk_token_target = int(compression_cfg.get("chunk_tokens", 10_000))

    if protect_last_tokens is None:
        protect_last_tokens = chunk_token_target * 2

    # Resolve protected ranges against the live view.
    live_hashes = [message_hash(m) for m in messages]
    protected_ranges = store.find_ranges_in_view(live_hashes) if store else []

    # Size-pressure default policy: try folding non-doc content first;
    # only escalate to docs when no non-doc chunk is available. We are
    # only in compress_step because should_compress fired, so if there
    # are no non-doc chunks to fold, the tail IS under pressure and
    # must eat into a document.
    #
    # Callers can override with an explicit ``is_eligible`` predicate
    # (e.g., a live-test scenario that wants to force doc folding).
    doc_policy = (cfg.get("documents") or {}).get("eligibility", "size_pressure")

    weights_cfg = cfg.get("compression", {}).get("weights")

    if is_eligible is not None:
        # Explicit caller-provided policy wins.
        chunk_start, chunk_end = _pick_chunk_bounds(
            messages, protect_first_n, chunk_token_target, protect_last_tokens,
            protected_ranges=protected_ranges,
            is_eligible=is_eligible,
            weights_cfg=weights_cfg,
        )
    else:
        # Pass 1: non-doc only.
        chunk_start, chunk_end = _pick_chunk_bounds(
            messages, protect_first_n, chunk_token_target, protect_last_tokens,
            protected_ranges=protected_ranges,
            is_eligible=lambda r: False,
            weights_cfg=weights_cfg,
        )
        # Pass 2: if nothing and policy allows auto-doc-fold, retry.
        if chunk_start >= chunk_end and doc_policy == "size_pressure":
            logger.info(
                "autobio compress_step: no non-doc chunk; escalating to doc fold",
            )
            chunk_start, chunk_end = _pick_chunk_bounds(
                messages, protect_first_n, chunk_token_target, protect_last_tokens,
                protected_ranges=protected_ranges,
                is_eligible=lambda r: True,
                weights_cfg=weights_cfg,
            )
        # doc_policy == "manual" never auto-folds documents. Only
        # explicit CM ops can trigger a doc fold in that mode.

    if chunk_start >= chunk_end:
        logger.debug("autobio compress_step: no eligible chunk")
        return CompressResult(messages, None, 0, 0, 0)

    chunk_tokens = estimate_messages_tokens_rough(messages[chunk_start:chunk_end])

    # Identify whether this chunk is the top of a chronological doc.
    # If so, we'll need to advance the pin's start_hash (or retire it)
    # once the fold completes. Monolithic docs that fold entirely are
    # also retired.
    doc_pin = _range_starting_at(protected_ranges, chunk_start)
    doc_context: Optional[Dict[str, Any]] = None
    if doc_pin is not None and doc_pin["kind"].startswith("doc_"):
        meta = doc_pin.get("meta") or {}
        doc_context = {
            "pin_id": doc_pin.get("pin_id"),
            "style": doc_pin["kind"].replace("doc_", ""),
            "title": meta.get("title") if isinstance(meta, dict) else None,
            "original_tokens": meta.get("original_tokens") if isinstance(meta, dict) else None,
        }

    summary_id = summarize_chunk(
        messages, chunk_start, chunk_end, store, cfg, model_info,
        doc_context=doc_context,
    )
    if summary_id is None:
        return CompressResult(messages, None, chunk_tokens, 0, 0)

    new_messages = fold_chunk(messages, chunk_start, chunk_end, summary_id, store)

    # Doc-pin bookkeeping: advance or retire the pin based on how much
    # of it we just folded.
    if doc_pin is not None and doc_pin["kind"].startswith("doc_"):
        pin_id = int(doc_pin["pin_id"])
        end_idx = int(doc_pin["end_index"])
        if chunk_end > end_idx:
            # Whole doc folded (monolithic, or chronological that
            # happened to cover the full remainder).
            store.remove_pin(pin_id)
            logger.info("autobio: doc pin %s fully folded — retired", pin_id)
        else:
            # Chronological partial fold: advance start to new top.
            new_top = messages[chunk_end]
            new_top_hash = message_hash(new_top)
            store.advance_pin_start(pin_id, new_top_hash)
            logger.info(
                "autobio: doc pin %s top advanced to %s", pin_id, new_top_hash,
            )

    summary = store.get_summary(summary_id) or {}
    summary_tokens = int(summary.get("tokens", 0) or 0)
    saved = chunk_tokens - summary_tokens
    return CompressResult(new_messages, summary_id, chunk_tokens, summary_tokens, saved)


def compress_step_archive(
    store: Store,
    cfg: Dict[str, Any],
    model_info: ModelInfo,
    *,
    protect_first_n: int = 1,
    protect_last_tokens: Optional[int] = None,
    is_eligible: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> CompressResult:
    """Archive-driven compress_step. Picks the next chunk by walking
    archive position order directly, summarizes it, marks the source
    messages covered. No live-list mutation — the caller renders
    afterwards via ``render_view``.

    The picker (``_pick_archive_chunk``) walks the archive in a single
    pass, accumulating an uncovered run until either it reaches
    ``chunk_token_target`` or hits a pin/doc/covered boundary. The
    chunk it returns is *always* position-contiguous by construction —
    no filtered list, no index translation, no per-run
    ``protect_first_n`` carve-out.

    The legacy list-based ``compress_step`` is kept for callers that
    still pass a messages list (older tests; documentation examples).
    """
    compression_cfg = cfg.get("compression", {})
    chunk_token_target = int(compression_cfg.get("chunk_tokens", 10_000))
    if protect_last_tokens is None:
        protect_last_tokens = chunk_token_target * 2
    weights_cfg = compression_cfg.get("weights")
    doc_policy = (cfg.get("documents") or {}).get("eligibility", "size_pressure")

    pick_kwargs = dict(
        chunk_token_target=chunk_token_target,
        protect_last_tokens=protect_last_tokens,
        protect_first_n=protect_first_n,
        weights_cfg=weights_cfg,
    )

    if is_eligible is not None:
        chunk_hashes, doc_context = _pick_archive_chunk(
            store, **pick_kwargs, is_eligible=is_eligible,
        )
    else:
        # Pass 1: non-doc only.
        chunk_hashes, doc_context = _pick_archive_chunk(
            store, **pick_kwargs, is_eligible=lambda r: False,
        )
        # Pass 2: if nothing and policy allows, retry with docs eligible.
        if not chunk_hashes and doc_policy == "size_pressure":
            logger.info(
                "autobio compress_step_archive: no non-doc chunk; "
                "escalating to doc fold",
            )
            chunk_hashes, doc_context = _pick_archive_chunk(
                store, **pick_kwargs, is_eligible=lambda r: True,
            )

    if not chunk_hashes:
        logger.debug("autobio compress_step_archive: no eligible chunk")
        return CompressResult([], None, 0, 0, 0)

    # Materialize the chunk's message dicts for token estimate +
    # summarizer prompt. summarize_chunk takes (messages, start, end).
    chunk_msgs = _materialize_chunk(store, chunk_hashes)
    chunk_tokens = estimate_messages_tokens_rough(chunk_msgs)

    summary_id = summarize_chunk(
        chunk_msgs, 0, len(chunk_msgs), store, cfg, model_info,
        doc_context=doc_context,
    )
    if summary_id is None:
        return CompressResult([], None, chunk_tokens, 0, 0)

    # Archive-only fold: mark the source messages as covered. The next
    # render_view() call will see the marker in their place.
    store.mark_compressed(chunk_hashes, summary_id)

    # Doc-pin bookkeeping: advance start_hash past the folded slice or
    # retire the pin entirely. Pins live by hash; we no longer have
    # positional indices to compare against, so we walk the archive.
    if doc_context is not None and doc_context.get("pin_id") is not None:
        _retire_or_advance_doc_pin(
            store, int(doc_context["pin_id"]), chunk_hashes,
        )

    summary = store.get_summary(summary_id) or {}
    summary_tokens = int(summary.get("tokens", 0) or 0)
    saved = chunk_tokens - summary_tokens
    return CompressResult([], summary_id, chunk_tokens, summary_tokens, saved)


def _materialize_chunk(
    store: Store, chunk_hashes: List[str],
) -> List[Dict[str, Any]]:
    """Convert chunk hashes back to OpenAI-format messages, preserving
    order. Used to feed the summarizer prompt builder."""
    rows = store.get_messages_by_hashes(chunk_hashes)
    return [_archive_row_to_msg(r) for r in rows]


def _retire_or_advance_doc_pin(
    store: Store, pin_id: int, folded_chunk_hashes: List[str],
) -> None:
    """After folding part of a doc pin, either retire the pin (if the
    fold consumed everything from start through end) or advance its
    ``start_hash`` to the first uncovered message past the fold.

    Uses hash-based pin metadata (``start_hash``, ``end_hash``) — no
    positional indices, since the picker no longer hands them back.
    """
    pins = [p for p in store.list_pins() if int(p["id"]) == pin_id]
    if not pins:
        return
    pin = pins[0]
    end_hash = pin.get("end_hash")
    if end_hash in folded_chunk_hashes:
        store.remove_pin(pin_id)
        logger.info("autobio: doc pin %s fully folded — retired", pin_id)
        return

    last_folded = folded_chunk_hashes[-1]
    with store._lock:
        cur = store._conn.execute(
            "SELECT position FROM messages "
            "WHERE session_id = ? AND hash = ?",
            (store.session_id, last_folded),
        ).fetchone()
    if cur is None:
        return
    last_pos = int(cur["position"] or 0)
    with store._lock:
        nxt = store._conn.execute(
            "SELECT hash FROM messages "
            "WHERE session_id = ? AND position > ? "
            "AND compressed_into IS NULL "
            "ORDER BY position ASC LIMIT 1",
            (store.session_id, last_pos),
        ).fetchone()
    if nxt is None:
        store.remove_pin(pin_id)
        logger.info(
            "autobio: doc pin %s consumed (no uncovered tail) — retired",
            pin_id,
        )
        return
    new_start = nxt["hash"]
    store.advance_pin_start(pin_id, new_start)
    logger.info("autobio: doc pin %s top advanced to %s", pin_id, new_start)


def _pick_archive_chunk(
    store: Store,
    *,
    chunk_token_target: int,
    protect_last_tokens: int,
    protect_first_n: int = 1,
    weights_cfg: Optional[Dict[str, Any]] = None,
    is_eligible: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Walk archive position order directly; emit one position-
    contiguous chunk of uncovered, unpinned messages.

    Returns ``(chunk_hashes, doc_context_or_None)``. Empty hashes mean
    "nothing to do this step" — not enough content, all pinned, or
    every uncovered run is reserved as the protect_last_tokens tail.

    Position is the source of truth. Because we walk in position
    order and emit a contiguous slice of one run, the resulting chunk
    is *always* position-contiguous — no filtered list, no per-run
    ``protect_first_n``, no risk of straddling a gap.

    State machine:

    - We accumulate ``current_run`` while we're in an uncovered,
      unpinned stretch.
    - On any boundary (covered row, plain pin, ineligible doc, end of
      archive), we close the run and ask: is the run foldable? If so,
      emit it and stop. Otherwise reset and continue.
    - Eligible docs short-circuit the walk and are returned directly.

    "Foldable" means the run has at least ``chunk_token_target``
    weighted tokens, with a tail-reservation of ``protect_last_tokens``
    deducted only if the run reaches end-of-archive.
    """
    eligible = is_eligible or (lambda r: False)
    all_rows = _all_messages_ordered(store)
    if not all_rows:
        return ([], None)

    pin_spans = _resolve_pin_spans(store, all_rows)

    n = len(all_rows)
    # current_run carries (idx, row, msg_dict, weighted_tokens) so the
    # closer has everything it needs without re-querying.
    current_run: List[Tuple[int, Dict[str, Any], Dict[str, Any], float]] = []

    def close_and_maybe_emit(at_end: bool) -> Optional[List[str]]:
        if not current_run:
            return None
        return _emit_archive_run(
            current_run, chunk_token_target,
            protect_last_tokens if at_end else 0,
            weights_cfg,
        )

    i = 0
    while i < n:
        row = all_rows[i]
        pos = int(row.get("position", i) or 0)

        # protect_first_n: only at the very head (position-based, not
        # per-run). Skipping never closes the run because at this
        # boundary the run is empty by construction.
        if pos < protect_first_n:
            i += 1
            continue

        pin = _pin_at_index(pin_spans, i)
        if pin is not None:
            kind = pin["kind"]
            if kind == "pin" or (
                kind.startswith("doc_") and not eligible(pin)
            ):
                emitted = close_and_maybe_emit(at_end=False)
                if emitted is not None:
                    return (emitted, None)
                current_run = []
                i = int(pin["end_index"]) + 1
                continue
            if kind == "doc_monolithic":
                doc_hashes = [
                    r["hash"]
                    for r in all_rows[pin["start_index"]:pin["end_index"] + 1]
                    if r.get("compressed_into") is None
                ]
                if doc_hashes:
                    return (doc_hashes, _doc_context_from_pin(pin))
                i = int(pin["end_index"]) + 1
                continue
            if kind == "doc_chronological":
                doc_hashes = []
                doc_tokens = 0.0
                for r in all_rows[pin["start_index"]:pin["end_index"] + 1]:
                    if r.get("compressed_into") is not None:
                        continue
                    msg = _archive_row_to_msg(r)
                    doc_hashes.append(r["hash"])
                    doc_tokens += _weighted_msg_tokens(msg, weights_cfg)
                    if doc_tokens >= chunk_token_target:
                        break
                if doc_hashes:
                    return (doc_hashes, _doc_context_from_pin(pin))
                i = int(pin["end_index"]) + 1
                continue
            # Unknown kind — defensive skip.
            i = int(pin["end_index"]) + 1
            continue

        if row.get("compressed_into") is not None:
            emitted = close_and_maybe_emit(at_end=False)
            if emitted is not None:
                return (emitted, None)
            current_run = []
            i += 1
            continue

        msg = _archive_row_to_msg(row)
        current_run.append((
            i, row, msg, _weighted_msg_tokens(msg, weights_cfg),
        ))
        i += 1

    # End of archive: try one more emit, this time honoring
    # protect_last_tokens.
    emitted = close_and_maybe_emit(at_end=True)
    if emitted is not None:
        return (emitted, None)
    return ([], None)


def _emit_archive_run(
    run: List[Tuple[int, Dict[str, Any], Dict[str, Any], float]],
    chunk_token_target: int,
    protect_last_tokens: int,
    weights_cfg: Optional[Dict[str, Any]],
) -> Optional[List[str]]:
    """Decide whether ``run`` is foldable; if so return chunk hashes.

    The run is a list of (idx, row, msg, weighted_tokens) tuples in
    archive position order.

    Foldability:

    - If ``protect_last_tokens > 0``, trim trailing messages whose
      cumulative weighted tokens reach that reserve. (Only at
      end-of-archive — the closer passes 0 for non-tail runs.)
    - The remaining prefix must total at least ``chunk_token_target``
      weighted tokens.
    - We emit the smallest prefix slice whose cumulative weighted
      tokens reach the target, then back-align to avoid splitting an
      assistant + tool_results group.

    Returns ``None`` if the run isn't foldable as-is.
    """
    if not run:
        return None

    usable_end = len(run)
    if protect_last_tokens > 0:
        reserved = 0.0
        j = len(run) - 1
        while j >= 0 and reserved < protect_last_tokens:
            reserved += run[j][3]
            j -= 1
        usable_end = j + 1
    if usable_end <= 0:
        return None

    usable = run[:usable_end]
    total_tokens = sum(t for _, _, _, t in usable)
    if total_tokens < chunk_token_target:
        return None

    cum = 0.0
    cut = 0
    for j, (_, _, _, t) in enumerate(usable):
        cum += t
        cut = j + 1
        if cum >= chunk_token_target:
            break

    # Forward-align: if the chunk has pending tool_calls (assistant
    # tool_calls without matching tool_results inside the chunk),
    # extend the cut forward to include the tool_results. Without
    # this, an assistant whose args alone exceed chunk_token_target
    # produces a single-message chunk with dangling calls — and the
    # subsequent ``_coerce_well_formed`` step drops the dangling
    # call, losing the conversational link.
    aligned = _align_chunk_msgs_forward(usable, cut)
    if aligned <= 0:
        return None

    return [r["hash"] for _, r, _, _ in usable[:aligned]]


def _align_chunk_msgs_forward(
    usable: List[Tuple[int, Dict[str, Any], Dict[str, Any], float]],
    cut: int,
) -> int:
    """Extend ``cut`` forward through ``usable`` to satisfy any tool
    calls inside the chunk that don't yet have their tool_results.

    Algorithm:
      1. Walk ``usable[:cut]`` once, tracking pending tool_call_ids
         (added when an assistant's tool_calls land, removed when a
         matching tool result lands).
      2. While pending is non-empty and there are more elements in
         ``usable``: if the next element is a tool message, include
         it (and clear its id from pending if matching); if it's
         non-tool, stop — the conversation moved on without
         resolving, which is an orphan we can't fix here.
      3. Return the (possibly larger) ``cut``.

    This replaces the older backward-cut alignment, which had a
    pathological case: a single assistant whose tool_calls'
    arguments alone exceeded ``chunk_token_target`` would hit the
    target on its own, then backward-cut would shrink the chunk to
    zero (because "the only msg is asst+tool_calls").
    """
    if cut <= 0:
        return cut
    pending: set = set()
    for _, _, m, _ in usable[:cut]:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid:
                    pending.add(cid)
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                pending.discard(cid)
    if not pending:
        return cut

    j = cut
    n = len(usable)
    while pending and j < n:
        m = usable[j][2]
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                pending.discard(cid)
            j += 1
            continue
        # Hit a non-tool while we still have pending calls. The
        # transcript itself has the orphan (no result for those
        # tool_calls in the original conversation). Stop here;
        # ``_coerce_well_formed`` in the summarizer prompt builder
        # drops unpaired tool_calls so the LLM call stays valid.
        break
    return j


def _all_messages_ordered(store: Store) -> List[Dict[str, Any]]:
    """All archived messages for this session, in position order.

    Replaces the old ``_uncovered_messages`` + ``_latest_contiguous_run``
    pair: the picker now walks the full archive and decides per-row
    whether to extend the current run or close it.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY position ASC, turn ASC",
            (store.session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _resolve_pin_spans(
    store: Store, all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve active pins to positional spans within ``all_rows``.

    ``find_ranges_in_view`` was originally designed for the live view
    (a partial subset of the archive); under the projection model we
    feed it the FULL archive's hash list so spans are stable across
    renders. Pins whose endpoints aren't both in the archive are
    silently dropped — same behavior as before.
    """
    live_hashes = [r["hash"] for r in all_rows]
    return store.find_ranges_in_view(live_hashes)


def _pin_at_index(
    pin_spans: List[Dict[str, Any]], idx: int,
) -> Optional[Dict[str, Any]]:
    """Return the pin span covering archive index ``idx``, or None."""
    for span in pin_spans:
        if int(span["start_index"]) <= idx <= int(span["end_index"]):
            return span
    return None


def _doc_context_from_pin(pin: Dict[str, Any]) -> Dict[str, Any]:
    """Build the doc_context blob the summarizer wants for a doc fold."""
    meta = pin.get("meta") or {}
    return {
        "pin_id": pin.get("pin_id"),
        "style": pin["kind"].replace("doc_", ""),
        "title": meta.get("title") if isinstance(meta, dict) else None,
        "original_tokens": (
            meta.get("original_tokens") if isinstance(meta, dict) else None
        ),
    }


def _archive_row_to_msg(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI-format message dict from an archive row.

    Used by compress_step_archive to feed the picker / summarizer.
    Reuses the render helper to keep the conversion logic in one
    place — they need to produce IDENTICAL shapes so message hashes
    are stable across the picker path and the live-view path.
    """
    from .render import _msg_to_openai
    return _msg_to_openai(row)


def _range_starting_at(
    ranges: List[Dict[str, Any]], idx: int,
) -> Optional[Dict[str, Any]]:
    """Return the range whose start_index == idx (if any)."""
    for r in ranges:
        if int(r["start_index"]) == idx:
            return r
    return None


# Backward-compatible alias — keeps the old name usable during the
# refactor. Callers should migrate to compress_step.
compress_to_l1 = compress_step


def _find_marker_index(
    messages: List[Dict[str, Any]], summary_id: str,
) -> Optional[int]:
    """Return the index of the summary marker for ``summary_id``, or None.

    Recognizes both the new tiny-marker prefix ``[autobio:memory`` and
    the legacy ``[memory L`` prefix so callers that still hold a
    pre-refactor live list (e.g., older tests) keep working.
    """
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if summary_id not in content:
            continue
        if "[autobio:memory" in content or "[memory L" in content:
            return i
    return None


# ---------------------------------------------------------------------------
# Boundary snapping (for future sub-message doc splitting)
# ---------------------------------------------------------------------------

# Regex fragments ordered by priority — earlier matches preferred when
# multiple candidates fall within the snap window.
import re as _re

_BOUNDARY_PATTERNS = [
    (_re.compile(r"\n\n+"),                              "paragraph"),
    (_re.compile(r"\n\[\d{4}[-/]\d{2}[-/]\d{2}"),        "iso_date"),
    (_re.compile(r"\n\[\d{2}:\d{2}:\d{2}"),              "timestamp"),
    (_re.compile(r"\n#{1,6}\s"),                          "md_header"),
    (_re.compile(r"\n[-=]{3,}\s*\n"),                     "rule"),
]


def snap_to_natural_boundary(
    text: str, target_offset: int, *, window: int = 500,
) -> int:
    """Return a character offset near ``target_offset`` that falls on a
    natural boundary (paragraph break, timestamp, header, rule).

    Currently unused — the message-level chunk picker already lands on
    clean boundaries (messages are themselves the unit). This helper is
    in place for a future extension that splits long tool results into
    sub-chunks at ingress time. See doc on single-message chronological
    documents in the plans file.
    """
    if not text:
        return target_offset
    lo = max(0, target_offset - window)
    hi = min(len(text), target_offset + window)
    window_text = text[lo:hi]
    best_offset = None
    best_distance = None
    for pattern, _label in _BOUNDARY_PATTERNS:
        for m in pattern.finditer(window_text):
            cand = lo + m.end()
            dist = abs(cand - target_offset)
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_offset = cand
        if best_offset is not None:
            break  # priority-ordered: first match wins
    return best_offset if best_offset is not None else target_offset


# ---------------------------------------------------------------------------
# L1→L2, L2→L3 merging (task 5)
# ---------------------------------------------------------------------------

def merge_if_ready(
    store: Store, cfg: Dict[str, Any], model_info: ModelInfo,
) -> List[str]:
    """Promote any level where ``merge_threshold`` position-contiguous
    unmerged summaries exist.

    A group of summaries is mergeable iff their archive-position
    coverage forms one contiguous run — no gaps, no other active
    summary or uncovered island between them. With pins, docs, or
    rollback creating uncovered islands between L1s, naive
    "by creation order" grouping produces L2s with non-contiguous
    coverage; ``render_view`` then emits one marker per contiguous
    run, breaking the "one summary == one marker" invariant. Checking
    position contiguity at merge time is the structural backstop.

    Runs up to max_levels-1 passes (L1→L2, L2→L3). Returns the list of
    newly created summary IDs. Safe to call repeatedly — on each call
    it only merges if a full contiguous group of ``merge_threshold``
    summaries is available.
    """
    compression_cfg = cfg.get("compression", {})
    threshold = int(compression_cfg.get("merge_threshold", 6))
    max_level = int(compression_cfg.get("max_levels", 3))
    target_tokens = int(compression_cfg.get("summary_target_tokens", 2_500))

    created: List[str] = []
    for level in range(1, max_level):
        while True:
            group = _find_contiguous_mergeable_group(store, level, threshold)
            if group is None:
                break
            batch = group[:threshold]
            new_id = _merge_batch(
                batch, level + 1, store, model_info, target_tokens,
                compression_cfg=compression_cfg,
            )
            if new_id is None:
                break
            created.append(new_id)
    return created


def _find_contiguous_mergeable_group(
    store: Store, level: int, threshold: int,
) -> Optional[List[Dict[str, Any]]]:
    """Return the first chronologically-ordered, position-contiguous
    group of >= ``threshold`` unmerged summaries at ``level``, or None.

    "Contiguous" means each summary's coverage is position-adjacent to
    the previous: ``prev_max_pos + 1 == next_min_pos``. Any gap (a
    pinned/uncovered island, a missing-source orphan, a non-adjacent
    rollback) breaks the group.

    Fallback: if NONE of the summaries have resolvable position
    ranges (e.g., a unit test inserting summaries without backing
    message rows, or a corrupt store where every source_hash is
    orphan), we fall back to creation-order grouping — the old
    behavior. In real operation every summary has backing messages
    with positions, so the contiguity check applies to all and the
    fallback never engages.
    """
    unmerged = store.list_summaries(level, include_merged=False)
    if len(unmerged) < threshold:
        return None

    sortable: List[Tuple[int, int, Dict[str, Any]]] = []
    unresolved: List[Dict[str, Any]] = []
    for s in unmerged:
        pos_min, pos_max = store.get_summary_position_range(s["id"])
        if pos_min is None or pos_max is None:
            unresolved.append(s)
            continue
        sortable.append((int(pos_min), int(pos_max), s))

    # If every summary at this level lacks positions, we're either in
    # a unit-test scenario (no backing messages) or a degraded state.
    # Fall back to creation-order so the merge mechanism still
    # exercises end-to-end. The position check is the safety net for
    # real data; with no positions to compare, we don't have a check
    # to apply and shouldn't block forever.
    if not sortable and unresolved:
        logger.warning(
            "merge_if_ready: no L%d summaries have resolvable positions; "
            "falling back to creation-order grouping (%d summaries)",
            level, len(unresolved),
        )
        if len(unresolved) >= threshold:
            return unresolved
        return None

    sortable.sort(key=lambda t: t[0])

    current: List[Dict[str, Any]] = []
    current_max: Optional[int] = None
    for pos_min, pos_max, s in sortable:
        if not current:
            current = [s]
            current_max = pos_max
            continue
        if current_max is not None and current_max + 1 == pos_min:
            current.append(s)
            current_max = pos_max
            continue
        # Gap → close current group. If it's already big enough,
        # return it; otherwise discard and start over.
        if len(current) >= threshold:
            return current
        current = [s]
        current_max = pos_max

    if len(current) >= threshold:
        return current
    return None


def _merge_batch(
    batch: List[Dict[str, Any]],
    new_level: int,
    store: Store,
    model_info: ModelInfo,
    target_tokens: int,
    *,
    compression_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Merge a batch of summaries into one at ``new_level``."""
    # Each source memory is presented as ``role=assistant`` because it
    # IS the agent's own first-person recall — putting agent-origin
    # content under ``user`` would tell the merge LLM "the user said
    # this happened" rather than "the agent recalls this happening."
    # The merge instruction itself stays ``user`` (it's a synthetic
    # task spec, not agent-origin content).
    prompt_messages: List[Dict[str, str]] = []
    for i, s in enumerate(batch, 1):
        prompt_messages.append({
            "role": "assistant",
            "content": f"MEMORY {i}/{len(batch)} ({s['id']}):\n{s['content']}",
        })
    prompt_messages.append({
        "role": "user",
        "content": _MERGE_INSTRUCTION.format(n=len(batch), target_tokens=target_tokens),
    })

    max_tokens_override = (compression_cfg or {}).get("max_tokens_override")
    merge_max_tokens = (int(max_tokens_override) if max_tokens_override
                        else int(target_tokens * 1.4))
    retries = int((compression_cfg or {}).get("summarize_retries", 3))
    member_source_hash = combined_hash([s["id"] for s in batch])
    summary_text = ""
    last_call_id: Optional[int] = None
    for attempt in range(1, retries + 1):
        summary_text, last_call_id = _call_summarizer(
            prompt_messages, model_info, max_tokens=merge_max_tokens,
            store=store, purpose=f"merge_l{new_level}",
            chunk_source_hash=member_source_hash,
        )
        if summary_text:
            if attempt > 1:
                logger.info(
                    "autobio L%d merge: succeeded on attempt %d/%d",
                    new_level, attempt, retries,
                )
            break
        logger.warning(
            "autobio L%d merge: empty output on attempt %d/%d (batch size %d)",
            new_level, attempt, retries, len(batch),
        )
    if not summary_text:
        logger.error(
            "autobio L%d merge: empty after %d retries for batch %s — halting",
            new_level, retries, [s["id"] for s in batch],
        )
        raise SummarizationFailed(
            member_source_hash, level=new_level,
            reason=f"empty merge output after {retries} retries",
        )

    # Combined source tracking: members' IDs serve as the "source_hashes"
    # at this level — uncompression walks them to rebuild the children.
    member_ids = [s["id"] for s in batch]
    parent_gen = max(int(s["compression_generations"]) for s in batch)

    # All ultimate-origin hashes concatenated — used for speculative match
    # if the same group of summaries is re-attempted.
    origin_hashes = []
    for s in batch:
        try:
            origin_hashes.extend(json.loads(s.get("source_hashes") or "[]"))
        except json.JSONDecodeError:
            pass

    new_id = store.insert_summary(
        level=new_level,
        source_hashes=member_ids,                       # direct children
        source_hash=combined_hash(origin_hashes or member_ids),
        content=summary_text,
        tokens=max(1, len(summary_text) // _CHARS_PER_TOKEN),
        compression_generations=parent_gen + 1,
        tags=None,
    )
    if last_call_id is not None and new_id is not None:
        try:
            store.attach_summary_to_llm_call(last_call_id, new_id)
        except Exception as e:
            logger.debug("attach_summary_to_llm_call (merge) failed: %s", e)
    store.mark_summary_merged(member_ids, new_id)
    logger.info(
        "autobio L%d merge: %s <- %s (gen %d)",
        new_level, new_id, member_ids, parent_gen + 1,
    )
    return new_id


# ---------------------------------------------------------------------------
# Internal: chunk selection
# ---------------------------------------------------------------------------

def _pick_chunk_bounds(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    chunk_token_target: int,
    protect_last_tokens: int,
    *,
    protected_ranges: Optional[List[Dict[str, Any]]] = None,
    is_eligible: Optional[Callable[[Dict[str, Any]], bool]] = None,
    weights_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """Pick the next chunk to fold, respecting protected ranges.

    Walks forward from ``protect_first_n`` looking for either:

    - a region of ordinary content (chunk_token_target worth), or
    - an eligible document whose chunk can be folded

    Protected ranges intervene:

    - Plain pins → always skipped (their end_index + 1 becomes the new
      cursor)
    - Non-eligible docs → skipped
    - Eligible ``doc_chronological`` → return a chunk from the doc's
      top edge, clamped to at most chunk_token_target
    - Eligible ``doc_monolithic`` → return the entire doc range

    ``protected_ranges`` is a list of dicts in the shape returned by
    ``Store.find_ranges_in_view`` — sorted by start_index ascending is
    preferred but not required (we sort internally).

    ``is_eligible(range) -> bool`` decides whether a given document is
    currently foldable. Default: never (pins and docs always protect).
    Callers supply a real predicate based on pressure, age, or manual
    signals. Plain ``kind='pin'`` is always protected regardless of
    the predicate.

    Both ends align so the chunk doesn't split an assistant/tool_result
    group. When no range intervenes, at least ``protect_last_tokens``
    worth of messages stay in the tail.
    """
    n = len(messages)
    if n <= protect_first_n + 1:
        return (protect_first_n, protect_first_n)

    ranges = sorted(protected_ranges or [], key=lambda r: int(r["start_index"]))
    eligible = is_eligible or (lambda r: False)

    cursor = _align_forward(messages, protect_first_n)

    # Skip/fold leading protected ranges until we land on free content.
    while cursor < n:
        r = _range_at_index(ranges, cursor)
        if r is None:
            break
        kind = r["kind"]
        if kind == "pin":
            cursor = int(r["end_index"]) + 1
            continue
        # Documents.
        if not eligible(r):
            cursor = int(r["end_index"]) + 1
            continue
        if kind == "doc_monolithic":
            return (int(r["start_index"]), int(r["end_index"]) + 1)
        if kind == "doc_chronological":
            doc_start = int(r["start_index"])
            doc_end_exclusive = min(int(r["end_index"]) + 1, n)
            tokens = 0.0
            end = doc_start
            for i in range(doc_start, doc_end_exclusive):
                tokens += _weighted_msg_tokens(messages[i], weights_cfg)
                end = i + 1
                if tokens >= chunk_token_target:
                    break
            end = _align_backward(messages, end)
            if end <= doc_start:
                return (doc_start, doc_start)
            return (doc_start, end)
        # Unknown kind — skip defensively.
        cursor = int(r["end_index"]) + 1

    if cursor >= n:
        return (n, n)

    # Free-region walk. The next range (if any) acts as a hard stop so
    # we never cross a pin boundary in a single chunk.
    boundary = n
    for r in ranges:
        if int(r["start_index"]) > cursor:
            boundary = int(r["start_index"])
            break

    start = cursor
    tokens = 0.0
    end = start
    for i in range(start, boundary):
        tokens += _weighted_msg_tokens(messages[i], weights_cfg)
        end = i + 1
        if tokens >= chunk_token_target:
            break

    end = _align_backward(messages, end)

    # Reserve protect_last_tokens worth of tail (also weighted). Only
    # applies when the chunk ends near the end of the live list —
    # doc-internal cuts already obey a tighter bound above.
    if end >= boundary - 1:
        tail_tokens = sum(
            _weighted_msg_tokens(m, weights_cfg) for m in messages[end:]
        )
        while end > start and tail_tokens < protect_last_tokens:
            end -= 1
            tail_tokens += _weighted_msg_tokens(messages[end], weights_cfg)
        end = _align_backward(messages, end)

    if end <= start:
        return (start, start)
    return (start, end)


def _range_at_index(
    ranges: List[Dict[str, Any]], idx: int,
) -> Optional[Dict[str, Any]]:
    """Return the range covering ``idx``, or None. Assumes ranges are
    sorted by start_index but does not require non-overlap — returns
    the first match."""
    for r in ranges:
        if int(r["start_index"]) <= idx <= int(r["end_index"]):
            return r
    return None


def _align_forward(messages: List[Dict[str, Any]], idx: int) -> int:
    """Slide past tool-result messages so we don't start mid-group."""
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


def _align_backward(messages: List[Dict[str, Any]], idx: int) -> int:
    """Pull an end boundary back before an assistant+tool_calls group."""
    if idx <= 0 or idx >= len(messages):
        return idx
    check = idx - 1
    while check >= 0 and messages[check].get("role") == "tool":
        check -= 1
    if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
        idx = check
    return idx


# ---------------------------------------------------------------------------
# Internal: prompt construction
# ---------------------------------------------------------------------------

def _build_kv_preserving_prompt(
    *,
    messages: List[Dict[str, Any]],
    chunk_start: int,
    chunk_end: int,
    store: Store,
    target_tokens: int,
    doc_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build the summarizer prompt in strict as-of-the-chunk order.

    The chunk is the NEWEST thing in the summarizer's context. Nothing
    that happened after the chunk is visible here, so the memory is
    formed exactly as it would have been at the moment the chunk was
    live — no retrospective framing from events that came later.

    Compression is always chronological (oldest chunk first), so every
    existing summary in the store predates this chunk by construction;
    no timestamp filtering is needed.

    Order:
      1. L3 summaries (oldest first, as user messages)
      2. L2 summaries
      3. L1 summaries (existing)
      4. Head — messages before the chunk
      5. Marker — the only synthetic addition to the historical state
      6. Chunk — the slice being compressed
      7. Instruction

    Notice there is *no* tail_after_chunk. Including it would leak
    future information into the KV state and corrupt the as-of framing.
    """
    prompt: List[Dict[str, Any]] = []

    # Prior summaries, oldest level first. All predate the chunk because
    # compression is strictly chronological.
    #
    # role=assistant: these memories ARE the agent's first-person recall
    # of earlier conversation. Presenting them as ``user`` would tell
    # the summarizer "the user told me X happened" rather than "I
    # remember X happening" — a meaningful KV-level distortion. The
    # strict role-alternation rule no longer applies (Anthropic
    # accepts consecutive assistants), so we can be honest about
    # speaker identity here.
    for level in (3, 2, 1):
        for s in store.list_summaries(level, include_merged=False):
            prompt.append({
                "role": "assistant",
                "content": f"MEMORY {s['id']} (gen {s['compression_generations']}):\n{s['content']}",
            })

    # Head only — everything before the chunk. No tail_after_chunk.
    head = messages[:chunk_start]
    prompt.extend(_shallow_copy(head))

    # Marker — the in-band signal that a memory is about to form. The
    # only synthetic element the historical instance didn't see.
    prompt.append({"role": "user", "content": _MARKER})

    # The chunk — the newest thing in this prompt, exactly as the
    # historical instance experienced it.
    prompt.extend(_shallow_copy(messages[chunk_start:chunk_end]))

    # Instruction. Doc chunks get a tailored version emphasising
    # chronological framing and detail preservation.
    if doc_context is not None:
        instruction_text = _DOC_CHUNK_INSTRUCTION.format(
            title=doc_context.get("title") or f"pin-{doc_context.get('pin_id', '?')}",
            style=doc_context.get("style", "chronological"),
            original_tokens=doc_context.get("original_tokens") or "unknown",
            target_tokens=target_tokens,
        )
    else:
        instruction_text = _INSTRUCTION.format(target_tokens=target_tokens)
    prompt.append({"role": "user", "content": instruction_text})

    return _coerce_well_formed(prompt)


def _shallow_copy(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in msgs:
        copy = dict(m)
        content = copy.get("content")
        if isinstance(content, list):
            # Multimodal → flatten to text for the summarizer.
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            copy["content"] = "\n".join(t for t in text_parts if t)
        out.append(copy)
    return out


def _coerce_well_formed(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure the prompt is a valid OpenAI-format message sequence.

    The summarizer doesn't care about exact correctness, but some
    providers reject messages with tool-role rows that have no matching
    assistant tool_calls. We drop orphans on BOTH sides:

    - tool result with no matching assistant tool_call → drop the result
    - assistant message whose tool_calls have no result in this prompt →
      drop the unmatched tool_call entries (and the assistant entirely
      if that leaves it empty with no other content)

    Critical: we DO NOT inject synthetic stubs to fill gaps. Earlier
    versions injected a fake ``[result omitted in summary prompt]``
    tool message, which violated principle #5 ("no fabricated memories")
    — the summarizer could weave the fake stub into the real summary
    text and the resulting L1 would describe content that never existed.
    Dropping unpaired calls is honest; stubbing them is not.
    """
    # Pass 1: collect assistant tool_call IDs and tool-result IDs that
    # appear in this prompt.
    call_ids = set()
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid:
                    call_ids.add(cid)
    result_ids = {
        m.get("tool_call_id", "")
        for m in msgs
        if m.get("role") == "tool" and m.get("tool_call_id")
    }

    cleaned: List[Dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id", "")
            if cid not in call_ids:
                continue   # orphan result — drop
            cleaned.append(m)
            continue
        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if not tool_calls:
                cleaned.append(m)
                continue
            # Keep only tool_calls that have a matching result in scope.
            kept_calls = []
            for tc in tool_calls:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid in result_ids:
                    kept_calls.append(tc)
            if kept_calls:
                copy = dict(m)
                copy["tool_calls"] = kept_calls
                cleaned.append(copy)
            elif m.get("content"):
                # All tool_calls were unpaired but the message still
                # has narrative content; keep the content and drop the
                # tool_calls so the prompt stays valid.
                copy = dict(m)
                copy.pop("tool_calls", None)
                cleaned.append(copy)
            # else: drop the assistant entirely — it would be empty.
            continue
        cleaned.append(m)

    return cleaned


# ---------------------------------------------------------------------------
# Internal: LLM call
# ---------------------------------------------------------------------------

def _call_summarizer(
    messages: List[Dict[str, Any]],
    model_info: ModelInfo,
    *,
    max_tokens: int,
    store: Optional[Store] = None,
    purpose: str = "summarize",
    chunk_source_hash: Optional[str] = None,
) -> "Tuple[str, Optional[int]]":
    """Invoke the auxiliary LLM and return ``(content, llm_call_id)``.

    Uses the ``compression`` task so provider/model are read from
    ``auxiliary.compression`` config. Returns empty string on any
    failure — the caller decides whether to retry or skip.

    When ``store`` is provided, every call is recorded in the
    ``llm_calls`` table with the full prompt + response. The returned
    ``llm_call_id`` lets the caller backfill ``summary_id`` (or
    ``cm_event_id``) once the downstream record exists.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as e:
        logger.warning("autobio: cannot import agent.auxiliary_client: %s", e)
        if store is not None:
            try:
                store.record_llm_call(
                    purpose=purpose, prompt_messages=messages,
                    response_text=None, model=model_info.model,
                    chunk_source_hash=chunk_source_hash,
                    error=f"import-failed: {e}",
                )
            except Exception:
                pass
        return ("", None)

    call_kwargs: Dict[str, Any] = {
        "task": "compression",
        "messages": messages,
        "max_tokens": max_tokens,
        "main_runtime": {
            "model": model_info.model,
            "provider": model_info.provider,
            "base_url": model_info.base_url,
            "api_key": model_info.api_key,
            "api_mode": model_info.api_mode,
        },
    }
    if model_info.summary_model_override:
        call_kwargs["model"] = model_info.summary_model_override

    response_text = ""
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    err: Optional[str] = None
    try:
        response = call_llm(**call_kwargs)
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        response_text = content.strip()
        finish_reason = _coerce_str(getattr(choice, "finish_reason", None))
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = _coerce_int(getattr(usage, "prompt_tokens", None))
            completion_tokens = _coerce_int(getattr(usage, "completion_tokens", None))
    except Exception as e:
        logger.warning("autobio summarizer call failed: %s", e)
        err = str(e)

    call_id: Optional[int] = None
    if store is not None:
        try:
            call_id = store.record_llm_call(
                purpose=purpose, prompt_messages=messages,
                response_text=response_text or None,
                model=(model_info.summary_model_override or model_info.model),
                response_finish=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                chunk_source_hash=chunk_source_hash,
                error=err,
            )
        except Exception as rec_e:
            logger.error("autobio: failed to record llm_call: %s", rec_e)

    return (response_text, call_id)


# ---------------------------------------------------------------------------
# Internal: summary marker + tool pair sanitization
# ---------------------------------------------------------------------------

def _make_summary_marker(summary_id: str, summary: str, level: int) -> Dict[str, Any]:
    """Build the live-view marker for a summary.

    Tiny — ~50 tokens — because under the projection model the marker
    is regenerated on every render and the full L1 body lives in the
    archive (CM fetches via memory_read / UNCOMPRESS). Prior versions
    inlined the whole summary text, which caused the live tail to grow
    instead of shrink on fold and let the picker recursively fold its
    own output.

    Uses ``role=assistant`` because the marker stands in for content
    the AGENT itself produced (folded autobiographical memory).
    Anthropic dropped the strict role-alternation requirement, so we
    no longer need the ``role=user`` workaround to sidestep
    "two assistants in a row" rejections.

    The ``_autobio_marker`` flag lets reconcile paths recognize
    markers by schema; the prefix ``[autobio:memory`` is a fallback
    for round-trips that drop unknown keys.
    """
    from .render import _MARKER_FLAG
    tokens_hint = max(1, len(summary) // _CHARS_PER_TOKEN) if summary else 0
    return {
        "role": "assistant",
        "content": (
            f"[autobio:memory {summary_id} · L{level} · {tokens_hint}t · "
            f"call cm_manage_context to expand]"
        ),
        _MARKER_FLAG: True,
    }


def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove orphan tool results and inject stubs for orphan tool calls.

    Mirrors the built-in compressor's helper so the post-compression
    message list is always a valid OpenAI-format sequence.
    """
    surviving: set = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid:
                    surviving.add(cid)

    result_ids: set = set()
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                result_ids.add(cid)

    # Drop orphan results.
    orphans = result_ids - surviving
    if orphans:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphans)
        ]

    # Stub missing results.
    missing = surviving - result_ids
    if missing:
        patched: List[Dict[str, Any]] = []
        for m in messages:
            patched.append(m)
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid in missing:
                        patched.append({
                            "role": "tool",
                            "content": "[result rolled into a memory — ask CM to uncompress if needed]",
                            "tool_call_id": cid,
                        })
        messages = patched

    return messages


def _msg_tokens(msg: Dict[str, Any]) -> int:
    """Rough RAW token count for a single message — what the model
    actually consumes. Used for thresholds and any place that needs
    real context-window accounting.
    """
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = json.dumps(content)
    tokens = len(content) // _CHARS_PER_TOKEN + 10
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            args = (tc.get("function") or {}).get("arguments", "") or ""
            tokens += len(args) // _CHARS_PER_TOKEN
    return tokens


def _weighted_msg_tokens(
    msg: Dict[str, Any], weights_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """Token count adjusted by per-role / per-tool weight.

    Used by the chunk picker to budget how much CONTENT to roll into
    one L1. Code and console output (Read/Bash/Grep/etc.) is heavily
    discounted because the L1 doesn't need to carry the literal bytes
    — CM can grep or re-read from disk; the L1 just has to remember
    "user explored module X". User and assistant text gets full weight
    because it only exists in the conversation.

    Tool name lookup is **hierarchical**: an ext-suffixed tool name
    (``Read:md``, set at archive time by ``tool_naming.enrich_tool_name``)
    is checked first; if not found, the bare prefix (``Read``) is
    checked; finally the ``tool_default`` falls back. This lets the
    weights config override per-extension (``Read:md: 0.5`` for
    documents) while keeping the bare ``Read: 0.1`` default for code.

    With ``weights_cfg=None``, this is identical to ``_msg_tokens``.
    Production sets weights via config; tests can pass an explicit
    dict.
    """
    raw = _msg_tokens(msg)
    if not weights_cfg:
        return float(raw)
    role = msg.get("role", "")
    if role == "tool":
        weight = _resolve_tool_weight(msg.get("name") or "", weights_cfg)
    else:
        weight = weights_cfg.get(
            role, weights_cfg.get("default", 1.0),
        )
    return float(raw) * float(weight)


def _resolve_tool_weight(
    name: str, weights_cfg: Dict[str, Any],
) -> float:
    """Resolve a tool's per-message weight with ext-suffix fallback.

    Lookup order:
      1. ``tool_by_name[name]`` — exact match on the (possibly
         ext-suffixed) name. E.g., ``Read:md`` overrides documents.
      2. ``tool_by_name[prefix]`` where prefix is everything before
         the first colon. E.g., a ``Read:tex`` we don't have a
         specific entry for falls back to plain ``Read``.
      3. ``tool_default`` — the umbrella default for tools the
         config doesn't mention at all.
    """
    by_name = weights_cfg.get("tool_by_name") or {}
    if name in by_name:
        return float(by_name[name])
    if ":" in name:
        prefix = name.split(":", 1)[0]
        if prefix in by_name:
            return float(by_name[prefix])
    return float(weights_cfg.get("tool_default", 0.5))


def _coerce_str(v: Any) -> Optional[str]:
    """Defensive: SQLite refuses to bind MagicMock and similar; only
    accept str|None. Useful when the LLM client returns mocked or
    odd-shaped objects in tests."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return None


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(v)
    except Exception:
        return None


def _missing_hashes(store: Store, hashes: List[str]) -> List[str]:
    """Return the subset of ``hashes`` that do NOT resolve in the
    messages table.

    Used to assert archive-first invariants before writing a summary —
    any hash referenced in source_hashes must already be archived, or
    the resulting summary cannot be honestly reconstructed later.
    """
    if not hashes or store is None:
        return []
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    placeholders = ",".join("?" * len(hashes))
    try:
        with store._lock:
            rows = conn.execute(
                f"SELECT hash FROM messages WHERE hash IN ({placeholders})",
                tuple(hashes),
            ).fetchall()
        present = {r[0] if not isinstance(r, dict) else r["hash"] for r in rows}
    except Exception as e:
        logger.error("autobio _missing_hashes: query failed: %s", e)
        # Conservative: treat as all-missing so the caller raises.
        return list(hashes)
    return [h for h in hashes if h not in present]

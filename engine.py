"""AutobioContextEngine — the compression + CM half of the plugin.

V1 skeleton: wires the lifecycle (session start/end, model update),
archives messages into the shared store before the live list gets
mutated, and delegates the actual compression + CM work to the modules
in this package (to be filled in by subsequent tasks).

Everything not yet implemented returns the message list unchanged so
the engine is safe to enable while under construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

from .config import load_autobio_config
from .ids import message_hash
from .render import is_marker_message, known_marker_hashes, render_view
from .store import Store, close_store, get_store

logger = logging.getLogger(__name__)

# Module-level registry so the MemoryProvider half can find the live
# engine for its session and tick speculation without going through
# run_agent.py plumbing.
_ACTIVE_ENGINES: Dict[str, "AutobioContextEngine"] = {}


def get_active_engine(session_id: str) -> Optional["AutobioContextEngine"]:
    return _ACTIVE_ENGINES.get(session_id)


class AutobioContextEngine(ContextEngine):
    """Hierarchical autobiographical memory — compression + CM side."""

    # Preflight defaults; refined by update_model() once we know context_length.
    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    def __init__(self) -> None:
        self.cfg = load_autobio_config()
        self._session_id: Optional[str] = None
        self._hermes_home: Optional[str] = None
        self._store: Optional[Store] = None
        self._speculative = None  # SpeculativeWorker, created on session_start

        # Local turn counter. The MemoryProvider half writes
        # ``current_turn`` to the shared KV from its ``on_turn_start``
        # hook; the engine half mirrors that write from ``on_turn_end``
        # so a deployment that loads only the engine still gets correct
        # per-message turn numbers in the archive. With both halves
        # loaded the values converge — see ``on_turn_end``.
        self._engine_turn_count: int = 0

        # Model info — set by update_model().
        self.model: str = ""
        self.base_url: str = ""
        self.api_key: str = ""
        self.provider: str = ""
        self.api_mode: str = ""
        self.context_length: int = 0
        self.threshold_tokens: int = 0
        self.compression_count: int = 0

        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0

        # Set when a summarizer failure occurs. While True, compress()
        # refuses to run and should_compress*() return False, so we do
        # not loop on a broken summarizer. Must be explicitly cleared
        # via resume_compression() (e.g., after the user switches to a
        # working auxiliary model).
        self._compression_halted: bool = False
        self._halt_reason: Optional[str] = None

    # -- identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "autobio"

    # -- token / model tracking -----------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0) or 0
        self.last_completion_tokens = usage.get("completion_tokens", 0) or 0
        self.last_total_tokens = usage.get("total_tokens", 0) or 0

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = max(
            int(context_length * self.threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        logger.info(
            "autobio engine: model=%s context_length=%d threshold=%d",
            model, context_length, self.threshold_tokens,
        )

    # -- compression gate ------------------------------------------------

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        if self._compression_halted:
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens > 0

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if self._compression_halted:
            return False
        # Cheap rough check — same estimate used elsewhere.
        from agent.model_metadata import estimate_messages_tokens_rough
        if not self.threshold_tokens:
            return False
        return estimate_messages_tokens_rough(messages) >= self.threshold_tokens

    def resume_compression(self) -> None:
        """Clear the halt flag after a human/CM has addressed the cause.

        Intended for use after the user switches auxiliary models, tops
        up API credits, or otherwise resolves whatever caused the
        summarizer to return empty. Does NOT retry pending compression
        — the next natural threshold trigger will.
        """
        if self._compression_halted:
            logger.info(
                "autobio: compression halt cleared (was: %s)", self._halt_reason,
            )
        self._compression_halted = False
        self._halt_reason = None

    # -- main entry point ------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Archive-first, then prune, then run a compression step
        (summarize + fold), then promote L1→L2→L3 if merge thresholds
        are met. Finally schedule speculation for the next compression.

        The speculative worker may have already written an L1 to the
        archive for the current chunk; compress_step picks it up via
        source_hash lookup without a second LLM call.
        """
        from .compression import (
            ModelInfo, SummarizationFailed, compress_step_archive,
            merge_if_ready,
        )
        from .tool_pruning import prune_tool_results

        # If a prior summarizer failure set the halt flag, do nothing.
        # Caller must resume_compression() before we try again.
        if self._compression_halted:
            return messages

        # Step 1: reconcile incoming messages with the archive. Real new
        # messages get archived with monotonic positions; stale render
        # markers from prior compress() calls are skipped.
        self._reconcile_with_archive(messages)

        # Step 2: tool pruning runs on a RENDERED view (not the input
        # `messages`, which may contain stale markers). Pruning still
        # operates on a live list because that's the right granularity
        # for "what the model sees" — but the list is fresh.
        rendered = render_view(
            self._store,
            protect_first_n=self.protect_first_n,
            do_not_render_as_marker=self._unfolded_summaries(),
        ) if self._store is not None else list(messages)

        current_turn = self._current_turn()
        pruned, stats = prune_tool_results(
            rendered, self._store, self.cfg["tool_pruning"],
            current_turn=current_turn,
        )
        if stats.dedup_count or stats.summarized_count or stats.truncated_args_count:
            logger.info(
                "autobio prune: dedup=%d summarized=%d args_trunc=%d archived=%d",
                stats.dedup_count, stats.summarized_count,
                stats.truncated_args_count, stats.archived_count,
            )

        if self._store is None:
            self.compression_count += 1
            return pruned

        model_info = ModelInfo(
            model=self.model, provider=self.provider,
            base_url=self.base_url, api_key=self.api_key,
            api_mode=self.api_mode,
        )

        # Step 3: archive-only compression. compress_step_archive picks
        # a chunk of UNCOVERED messages from the archive (not from a
        # live list), summarizes, and marks them covered. No live-list
        # mutation.
        try:
            compress_step_archive(
                self._store, self.cfg, model_info,
                protect_first_n=self.protect_first_n,
            )
            merged = merge_if_ready(self._store, self.cfg, model_info)
            if merged:
                logger.info("autobio merged: %s", merged)
        except SummarizationFailed as failure:
            # Halt. Do NOT fabricate a placeholder — the archive is
            # authoritative and every summary must be a real memory the
            # summarizer actually produced. Surface the failure via a
            # visible assistant message so the user/CM can see the
            # problem. All future compress() / preflight checks
            # short-circuit until resume_compression() is called.
            self._compression_halted = True
            self._halt_reason = str(failure)
            halt_view = render_view(
                self._store,
                protect_first_n=self.protect_first_n,
                do_not_render_as_marker=self._unfolded_summaries(),
            )
            halt_view.append(self._build_halt_message(failure, halt_view))
            logger.error(
                "autobio: COMPRESSION HALTED — %s. Injected visible halt "
                "message; resume with resume_compression() after fixing.",
                failure,
            )
            self.compression_count += 1
            return halt_view

        # Step 4: render fresh from the post-compress archive state.
        new_messages = render_view(
            self._store,
            protect_first_n=self.protect_first_n,
            do_not_render_as_marker=self._unfolded_summaries(),
        )

        # Step 5: speculate the next compression.
        self.maybe_speculate(new_messages)

        self.compression_count += 1
        return new_messages

    def _reconcile_with_archive(self, messages: List[Dict[str, Any]]) -> None:
        """Archive any new content from the input messages.

        ``messages`` is what the runner believes is the current live
        list — including any stale render markers from a prior
        compress() call. We:

        - Skip markers (recognized by content prefix + verified
          against the set of hashes a fresh render would produce
          for currently-active summaries).
        - Enrich tool results with extension-suffixed tool names
          (see ``tool_naming.enrich_tool_name``) so weights resolve
          ``Read:md`` distinctly from ``Read:py``.
        - Hash and archive everything else, preserving monotonic
          positions.

        Idempotent on hash. Real new messages land in the archive with
        positions strictly larger than anything already there.
        """
        if not self._store:
            return
        from .tool_naming import build_tool_args_index, enrich_tool_name
        args_index = build_tool_args_index(messages)

        marker_hashes = known_marker_hashes(self._store)
        next_pos = self._next_position()
        current_turn = self._current_turn()
        for msg in messages:
            try:
                if is_marker_message(msg):
                    continue
                enriched = enrich_tool_name(msg, args_index)
                h = message_hash(enriched)
                if h in marker_hashes:
                    continue   # belt-and-braces: matches an active marker
                # If already archived, upsert is a no-op via INSERT OR
                # IGNORE. Position passed here is only used on first
                # insert; if the row exists it keeps its original.
                if self._store.get_message(h) is None:
                    self._store.upsert_message(
                        h, enriched, turn=current_turn, position=next_pos,
                    )
                    next_pos += 1
            except Exception as e:
                logger.error(
                    "autobio reconcile: archive FAILED for one message "
                    "(turn %s): %s", current_turn, e,
                )

    def _next_position(self) -> int:
        """Return the next monotonic position to assign on archival.

        Always queried fresh from MAX(position) — caching across
        compress() calls created a real bug: iter-1's reconcile would
        reuse iter-0's position counter from cache, producing
        duplicate positions and breaking the chunk picker's
        contiguous-run detection. Within a single reconcile loop the
        caller increments locally; the next reconcile re-queries.
        """
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT COALESCE(MAX(position), -1) FROM messages "
                "WHERE session_id = ?",
                (self._store.session_id,),
            ).fetchone()
        return (row[0] if row else -1) + 1

    def _unfolded_summaries(self) -> "set[str]":
        """Summaries the user/CM has explicitly unfolded — render as
        children rather than a single marker.

        Stored as a JSON list under KV key ``unfolded_summaries``.
        Empty by default.
        """
        if self._store is None:
            return set()
        raw = self._store.kv_get("unfolded_summaries", []) or []
        if isinstance(raw, list):
            return set(raw)
        return set()

    def _build_halt_message(
        self,
        failure: "Exception",
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compose the in-dialog message the main instance sees after halt."""
        from agent.model_metadata import estimate_messages_tokens_rough
        tail_tokens = estimate_messages_tokens_rough(messages)
        limit = self.context_length or 0
        threshold = self.threshold_tokens or 0
        body = (
            "[autobio compression HALTED]\n"
            f"Reason: {failure}\n"
            f"Current tail: ~{tail_tokens} tokens "
            f"(threshold {threshold}, context limit {limit}).\n"
            "The summarizer produced no output and I refuse to fabricate a "
            "placeholder memory — every summary must be a real memory the "
            "summarizer actually formed. Until this is resolved, no further "
            "compression will run; if the tail continues to grow, the main "
            "model call itself will eventually fail.\n"
            "To resolve: have the user or CM investigate (wrong auxiliary "
            "model, exhausted API credits, reasoning budget too tight, "
            "etc.), then call engine.resume_compression() or issue a CM "
            "RESUME_COMPRESSION op."
        )
        return {"role": "assistant", "content": body}

    def maybe_speculate(self, messages: List[Dict[str, Any]]) -> None:
        """Enqueue a speculative compression if conditions are met.

        Called by the engine itself after compress(), and by the
        MemoryProvider half per-turn when the tail is approaching the
        threshold. Idempotent: the worker dedupes by combined_hash.
        """
        if self._speculative is None:
            return
        self._speculative.schedule(messages, protect_first_n=self.protect_first_n)

    # -- session lifecycle ----------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        from .compression import ModelInfo
        from .speculative import SpeculativeWorker

        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
        self._store = get_store(
            self._hermes_home, session_id,
            subdir=self.cfg["store"]["subdir"],
        )
        self._speculative = SpeculativeWorker(
            self._store, self.cfg,
            ModelInfo(
                model=self.model, provider=self.provider,
                base_url=self.base_url, api_key=self.api_key,
                api_mode=self.api_mode,
            ),
            engine=self,
        )
        # Restore the local turn counter from the shared KV so a
        # resumed session (where messages may have been archived under
        # an earlier value) does not regress to 0.
        self._engine_turn_count = int(
            self._store.kv_get("current_turn", 0) or 0
        )
        _ACTIVE_ENGINES[session_id] = self
        logger.info("autobio engine: session=%s store=%s", session_id, self._store.db_path)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Final archive pass to catch anything added since last compress().
        # If any message failed to archive on the way out, log loudly —
        # those messages are now ONLY in memory and will be lost when the
        # process exits. This is the last chance to detect that loss.
        failures = self._archive_live(messages)
        if failures:
            logger.error(
                "autobio session_end: %d/%d messages FAILED final archival "
                "(session=%s) — content lost on close. Investigate "
                "preceding error logs.",
                failures, len(messages), session_id,
            )

        # Wait briefly for in-flight speculative work; logs from the
        # store close path catch any lingering writes.
        if self._speculative is not None:
            self._speculative.shutdown()
            self._speculative = None
        if self._hermes_home and self._session_id:
            close_store(self._hermes_home, self._session_id)
        _ACTIVE_ENGINES.pop(session_id, None)
        self._store = None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        # Leave the on-disk archive intact; a reset is just a fresh tail.
        # Catastrophic recovery (deferred to v2) would decide whether to
        # reload summaries here.

    # -- per-turn hook --------------------------------------------------

    def on_turn_end(self, messages: List[Dict[str, Any]]) -> None:
        """Run per-turn maintenance with full message-list visibility.

        - Archive every message currently in the live list with its
          correct turn number. Without this, speculative summarization
          would operate on un-archived messages, producing L1s with
          source_hashes that don't resolve.
        - Schedule speculative compression for the upcoming threshold,
          so when compress() fires the result is already cached.
        - Run a proactive CM check-in if one is due. CM now receives
          the actual live messages for decisioning; if it INJECTs, the
          returned assistant message is appended to ``messages`` in
          place so the next turn sees it as real conversation history
          rather than a deferred observation.
        """
        if self._store is None:
            return

        # Mirror the MemoryProvider half's per-turn ``current_turn``
        # write so the archive turn numbers stay correct when only the
        # engine half is loaded. Take the max with the existing KV
        # value so we converge with whatever the provider wrote in
        # ``on_turn_start`` rather than overwriting a higher value.
        self._engine_turn_count += 1
        existing = int(self._store.kv_get("current_turn", 0) or 0)
        if existing >= self._engine_turn_count:
            self._engine_turn_count = existing
        else:
            self._store.kv_set("current_turn", self._engine_turn_count)

        # Archive first — speculative worker relies on messages being
        # in the archive to resolve source_hashes correctly.
        self._archive_live(messages)

        # Speculate for the NEXT compression based on current state.
        self.maybe_speculate(messages)

        # Proactive CM check-in — now in-dialog (injected into the
        # shared live list so the main instance sees it next turn).
        injected = self.maybe_proactive_checkin(messages)
        if injected is not None:
            messages.append(injected)

    # -- tool schemas ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "cm_manage_context",
                "description": (
                    "Call the Context Manager to mediate your memory. Use for "
                    "recall ('what was that decision about auth?'), compression "
                    "hints ('starting a new topic, compress aggressively'), pin "
                    "requests ('keep the last 20 turns in full'), and any "
                    "memory-related question. CM has visibility into every "
                    "compressed layer and the full on-disk archive, which you "
                    "do not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "Your natural-language request to CM.",
                        },
                    },
                    "required": ["request"],
                },
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        import json as _json
        if name != "cm_manage_context":
            return _json.dumps({"error": f"autobio engine: unknown tool {name}"})
        from .cm import cm_respond
        request = (args or {}).get("request", "")
        if not isinstance(request, str) or not request.strip():
            return _json.dumps({"error": "cm_manage_context requires a non-empty 'request'"})
        messages = kwargs.get("messages") or []
        reply = cm_respond(request.strip(), self, messages=messages)
        return _json.dumps({"reply": reply})

    # -- internals -------------------------------------------------------

    def _archive_live(self, messages: List[Dict[str, Any]]) -> int:
        """Hash and archive every NON-MARKER message in the live list.

        Idempotent via ``INSERT OR IGNORE`` on the content hash. Stale
        render markers from prior compress() calls are recognized via
        :func:`is_marker_message` plus a hash check against
        ``known_marker_hashes`` and explicitly skipped — archiving them
        as messages would let the picker see them as foldable content,
        producing the recursive Torment Nexus the projection model
        was designed to prevent.

        Tool results are enriched with their resolved tool name (and
        for ``Read``-family tools, a ``:ext`` suffix derived from the
        file path) BEFORE hashing, so the picker's per-tool weight
        lookup can distinguish ``Read:md`` (a document — full weight)
        from ``Read:py`` (source code — discounted).

        Uses monotonic positions seeded from MAX(position) so positions
        survive renders. After archival, tool results over the
        configured threshold are auto-marked as documents.

        Returns the count of messages that FAILED to archive.
        """
        if not self._store:
            return 0
        from .tool_naming import build_tool_args_index, enrich_tool_name

        # Build the tool_call_id → {name, args} map from any assistant
        # tool_calls in the current view. Used to suffix Read tool
        # results with the file extension at archive time.
        args_index = build_tool_args_index(messages)

        marker_hashes = known_marker_hashes(self._store)
        next_pos = self._next_position()
        current_turn = self._current_turn()
        failure_count = 0
        for msg in messages:
            try:
                if is_marker_message(msg):
                    continue
                # Enrich tool results with extension-suffixed name so
                # the hash and the stored row agree (and so the picker
                # weight lookup picks up Read:md vs Read:py).
                enriched = enrich_tool_name(msg, args_index)
                h = message_hash(enriched)
                if h in marker_hashes:
                    continue   # belt-and-braces
                if self._store.get_message(h) is not None:
                    continue   # already archived
                self._store.upsert_message(
                    h, enriched, turn=current_turn, position=next_pos,
                )
                next_pos += 1
                self._maybe_auto_mark_document(h, enriched)
            except Exception as e:
                logger.error(
                    "autobio: archive FAILED (turn %s): %s",
                    current_turn, e,
                )
                failure_count += 1
        return failure_count

    def _maybe_auto_mark_document(
        self, hash_id: str, msg: Dict[str, Any],
    ) -> None:
        """If a tool result exceeds the auto-detect threshold, create a
        document pin covering it.

        Idempotent: a lightweight query checks whether a doc pin already
        starts at this hash; if so, nothing is done. Tool results listed
        in ``exclude_tools`` never auto-mark.
        """
        doc_cfg = self.cfg.get("documents") or {}
        if not doc_cfg.get("auto_detect", True):
            return
        if self._store is None or msg.get("role") != "tool":
            return

        content = msg.get("content")
        if not isinstance(content, str):
            return
        threshold = int(doc_cfg.get("auto_threshold_tokens", 50_000))
        # Rough token estimate (chars/4).
        est_tokens = len(content) // 4
        if est_tokens < threshold:
            return

        # The stored ``name`` may carry an ext suffix from
        # ``tool_naming.enrich_tool_name`` (e.g., ``read_file:log``).
        # For "what tool produced this document?" semantics — exclude
        # checks, title text, meta — we want the bare tool name.
        # The suffix is purely a picker hint, not tool identity.
        stored_name = msg.get("name") or ""
        tool_name = stored_name.split(":", 1)[0] if ":" in stored_name else stored_name
        exclude = doc_cfg.get("exclude_tools") or []
        if tool_name and tool_name in exclude:
            return

        # Idempotency: don't double-mark.
        existing = self._store.list_pins()
        if any(
            p["start_hash"] == hash_id
            and str(p.get("kind", "")).startswith("doc_")
            for p in existing
        ):
            return

        style = doc_cfg.get("auto_style", "chronological")
        kind = f"doc_{style}" if style in ("chronological", "monolithic") else "doc_chronological"
        title = (
            f"auto: {tool_name} ({est_tokens:,} tokens)"
            if tool_name else f"auto: tool result ({est_tokens:,} tokens)"
        )
        meta = {
            "source": "auto",
            "tool_name": tool_name,
            "tool_call_id": msg.get("tool_call_id"),
            "original_tokens": est_tokens,
            "title": title,
            "detected_at_turn": self._current_turn(),
        }
        pin_id = self._store.add_pin(
            hash_id, hash_id, reason=title, kind=kind, meta=meta,
        )
        logger.info(
            "autobio: auto-detected document pin %s for %s (%d tokens, kind=%s)",
            pin_id, tool_name or "tool", est_tokens, kind,
        )

    def _current_turn(self) -> int:
        if not self._store:
            return 0
        return int(self._store.kv_get("current_turn", 0) or 0)

    # -- proactive check-in plumbing ------------------------------------

    def checkin_due(self) -> bool:
        """True if enough turns have passed since the last proactive check-in."""
        if self._store is None:
            return False
        interval = int(self.cfg.get("cm", {}).get("checkin_interval", 50))
        deferred_until = int(self._store.kv_get("deferred_until_turn", 0) or 0)
        current = self._current_turn()
        if current < deferred_until:
            return False
        last = int(self._store.kv_get("last_checkin_turn", -interval) or -interval)
        return (current - last) >= interval

    def record_checkin(self, turn: int) -> None:
        if self._store is None:
            return
        self._store.kv_set("last_checkin_turn", turn)
        self._store.kv_set("turns_since_checkin", 0)
        # Deferred_until_turn is set by _defer_checkin_for and NOT reset
        # here — a record_checkin for INJECT doesn't cancel a DEFER that
        # was issued in the same call. (In practice only one fires.)

    def _defer_checkin_for(self, turns: int) -> None:
        if self._store is None:
            return
        current = self._current_turn()
        self._store.kv_set("deferred_until_turn", current + max(1, int(turns)))

    def maybe_proactive_checkin(
        self, messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """If due, run a proactive CM check-in and return the injected
        message (or None if CM deferred)."""
        if not self.checkin_due():
            return None
        from .cm import proactive_checkin
        msg = proactive_checkin(self, messages)
        # Record only when CM actually injected — a DEFER leaves the
        # check-in counter untouched so the scheduled retry runs at the
        # deferred turn.
        if msg is not None:
            self.record_checkin(self._current_turn())
        return msg

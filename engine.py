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
from .render import is_recall_message, known_recall_hashes, render_view
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
        # Clamp protect_last_tokens against the model's actual
        # threshold. Reserving most of the threshold as raw tail
        # leaves the chunk picker with no room to fold under
        # pressure. Cap at threshold/3, which preserves the
        # production default's tail-to-threshold ratio (250k/750k =
        # 1/3 on a 1M model) and degrades sanely on smaller models —
        # e.g., tests using 100k context get clamped to ~25k tail
        # rather than nullifying the picker.
        compression_cfg = self.cfg.setdefault("compression", {})
        cfg_plt = compression_cfg.get("protect_last_tokens")
        if cfg_plt is not None:
            cap = max(self.threshold_tokens // 3, 0)
            if int(cfg_plt) > cap > 0:
                logger.warning(
                    "autobio: protect_last_tokens=%d exceeds threshold/3=%d "
                    "for context_length=%d (threshold=%d); clamping.",
                    cfg_plt, cap, context_length, self.threshold_tokens,
                )
                compression_cfg["protect_last_tokens"] = cap
        logger.info(
            "autobio engine: model=%s context_length=%d threshold=%d "
            "protect_last_tokens=%s",
            model, context_length, self.threshold_tokens,
            compression_cfg.get("protect_last_tokens"),
        )

    # -- compression gate ------------------------------------------------

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        if self._compression_halted:
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens > 0

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """True if a preflight compress should fire before the next
        API call.

        Three independent trigger conditions, any of which forces
        the runner to call ``compress()``:

        1. **Token threshold hit**: ``messages`` weighs more than the
           configured threshold. Standard automatic-compression case.
        2. **Compression halt cleared and pending content**: skipped
           if halted; otherwise see (3).
        3. **Any active summary exists**: as long as the archive has
           memory the engine should own, fire so ``compress()`` can
           re-render the conversation from current archive state. This
           covers:

             - Transfer-in / resume: a session with seeded memory
               that has to be projected into conversation_history
               before the first API call.
             - Mid-session UNCOMPRESS / REFOLD: the
               ``unfolded_summaries`` KV set changed so the
               rendered view needs to switch a region between recall
               pair and raw messages.
             - Newly-formed L1s from the previous turn that aren't
               yet reflected in the runner's conversation_history.

        Re-rendering is cheap (a SQL walk + a few list ops). The
        actual cost-bearing work — calling the summarizer LLM —
        only fires inside ``compress_step_archive`` when the picker
        finds an eligible chunk. So firing preflight every turn for
        memory-bearing sessions has negligible cost beyond the
        per-turn render itself.
        """
        if self._compression_halted:
            return False

        from agent.model_metadata import estimate_messages_tokens_rough

        if self.threshold_tokens and (
            estimate_messages_tokens_rough(messages) >= self.threshold_tokens
        ):
            return True

        # Fire when archive state has anything the runner's
        # conversation_history doesn't reflect. Two cases:
        #
        # 1. Summaries / folds exist — render must surface them as
        #    recall pairs; UNCOMPRESS toggles must take effect.
        # 2. Transfer-in: raw archive seeded out-of-band (e.g., via
        #    the ingest CLI) but conversation_history is still empty
        #    or partial. Without this check, the agent boots with no
        #    context on the first turn. compress() then runs render-
        #    only (the pressure gate skips fold for no-pressure
        #    transfer-in cases), populating conversation_history
        #    from the archive before the first API call.
        if self._store is not None:
            if self._store.summary_counts():
                return True
            if self._store.message_count() > len(messages):
                return True

        return False

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
        **kwargs: Any,  # absorb runner-side hints (focus_topic, etc.)
    ) -> List[Dict[str, Any]]:
        """Archive-first, then prune, then maybe fold a chunk (only
        under genuine pressure), then re-render. Finally schedule
        speculation.

        The fold step (``compress_step_archive`` + ``merge_if_ready``)
        only runs when the rendered view's token estimate is at or
        above the configured threshold. Preflight may fire for several
        non-pressure reasons — transfer-in, UNCOMPRESS toggle, newly-
        speculated L1s in the archive — but in those cases we want to
        *re-render* the live view from current archive state without
        also inventing a new fold. Folding is reserved for "we're
        actually about to overflow context," not "compress() got
        called for any reason."
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

        # Step 3: archive-only compression — gated on actual pressure.
        # The picker can always *find* a chunk if there's content past
        # the protect_last_tokens tail, but folding when the live view
        # is comfortably below threshold disturbs KV state for no
        # reason. Only fold when the rendered+pruned view is at or
        # above threshold. merge_if_ready is gated together because
        # promoting speculative-only L1s into L2s breaks the source-
        # hash cache (find_summary_by_source_hash filters merged L1s).
        from agent.model_metadata import estimate_messages_tokens_rough
        view_tokens = estimate_messages_tokens_rough(pruned)
        under_pressure = (
            self.threshold_tokens > 0
            and view_tokens >= self.threshold_tokens
        )
        try:
            if under_pressure:
                compress_step_archive(
                    self._store, self.cfg, model_info,
                    protect_first_n=self.protect_first_n,
                )
                merged = merge_if_ready(self._store, self.cfg, model_info)
                if merged:
                    logger.info("autobio merged: %s", merged)
            else:
                logger.debug(
                    "autobio compress: render-only (view=%d < threshold=%d)",
                    view_tokens, self.threshold_tokens,
                )
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

        recall_hashes = known_recall_hashes(self._store)
        next_pos = self._next_position()
        current_turn = self._current_turn()
        for msg in messages:
            try:
                if is_recall_message(msg):
                    continue
                enriched = enrich_tool_name(msg, args_index)
                h = message_hash(enriched)
                if h in recall_hashes:
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

    # -- per-call hook --------------------------------------------------

    def on_llm_call(
        self,
        api_kwargs: Dict[str, Any],
        response: Any = None,
        *,
        error: Optional[Exception] = None,
    ) -> None:
        """Record the main agent's per-call request/response into the
        ``llm_calls`` archive table so the viewer can show exactly
        what prompt the agent saw on each turn.

        Without this, the viewer's LLM-call records cover only
        plugin-internal calls (summarizer / merge / CM); the main
        agent's actual prompts — which is where memory-context
        rendering matters most — are not persisted anywhere
        inspectable. With it, every model call shows up under
        ``purpose="main"`` with full prompt + response.

        Provider-shape handling: OpenAI-compatible responses expose
        ``choices[0].message.content``; Anthropic native exposes
        ``content[N].text``; Bedrock converse is normalized to the
        OpenAI shape upstream. We try OpenAI first, then fall back
        to Anthropic blocks.
        """
        if self._store is None:
            return
        try:
            messages = api_kwargs.get("messages") or []
            model = (
                api_kwargs.get("model")
                or self.model
                or "unknown"
            )

            if response is None or error is not None:
                self._store.record_llm_call(
                    purpose="main",
                    prompt_messages=messages,
                    response_text=None,
                    model=model,
                    error=str(error) if error else "no response",
                )
                return

            response_text: Optional[str] = None
            finish_reason: Optional[str] = None

            choices = getattr(response, "choices", None)
            if choices:
                try:
                    choice = choices[0]
                    msg = getattr(choice, "message", None)
                    if msg is not None:
                        response_text = getattr(msg, "content", None) or ""
                    finish_reason = getattr(choice, "finish_reason", None)
                except (AttributeError, IndexError):
                    pass

            if response_text is None:
                # Anthropic native: response.content is a list of blocks
                # (text / tool_use / thinking / …). Concatenate text.
                content_blocks = getattr(response, "content", None)
                if content_blocks:
                    pieces: List[str] = []
                    for blk in content_blocks:
                        text = getattr(blk, "text", None)
                        if text:
                            pieces.append(text)
                    response_text = "\n".join(pieces) if pieces else ""
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason:
                    finish_reason = str(stop_reason)

            usage = getattr(response, "usage", None)
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            if usage is not None:
                pt = (
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "input_tokens", None)
                )
                ct = (
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", None)
                )
                try:
                    prompt_tokens = int(pt) if pt is not None else None
                    completion_tokens = int(ct) if ct is not None else None
                except (TypeError, ValueError):
                    pass

            self._store.record_llm_call(
                purpose="main",
                prompt_messages=messages,
                response_text=response_text,
                model=str(model),
                response_finish=str(finish_reason) if finish_reason else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            logger.warning("autobio on_llm_call recording failed: %s", e)

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
        :func:`is_recall_message` plus a hash check against
        ``known_recall_hashes`` and explicitly skipped — archiving them
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

        recall_hashes = known_recall_hashes(self._store)
        next_pos = self._next_position()
        current_turn = self._current_turn()
        failure_count = 0
        for msg in messages:
            try:
                if is_recall_message(msg):
                    continue
                # Enrich tool results with extension-suffixed name so
                # the hash and the stored row agree (and so the picker
                # weight lookup picks up Read:md vs Read:py).
                enriched = enrich_tool_name(msg, args_index)
                h = message_hash(enriched)
                if h in recall_hashes:
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

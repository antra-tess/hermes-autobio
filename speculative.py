"""Speculative summarization worker.

When the live tail approaches the compression threshold, we run the
summarizer in a background thread so the L1 is already in the archive
by the time compression fires. Under the summarize/fold split, this
is *purely* an archive write — the live view is untouched until the
engine's compress_step decides to fold.

How reuse works:

- ``schedule`` picks the prospective chunk and kicks off
  ``summarize_chunk`` in a background thread.
- ``summarize_chunk`` is idempotent: it writes to the archive once and
  short-circuits on subsequent calls (via source_hash lookup).
- When ``compress_step`` later hits the same chunk, ``summarize_chunk``
  finds the pre-computed L1 and skips the LLM call.
- /undo or /retry that changes the chunk just means the real chunk
  has a different source_hash; the speculative L1 is still in the
  archive (harmless — it's a valid memory of a chunk that no longer
  exists in the live view), and a fresh summary gets written for the
  new chunk.

Design notes:

- Only one speculative summarization is in flight at a time. L1 #N+1
  depends on whether L1 #N was accepted, so parallelism here is
  unsafe.
- The worker writes to ``summaries`` like any other summarization.
  SQLite WAL mode handles the concurrent writer just fine.
- Snapshots are dict copies — the worker never reads the live list
  after kickoff.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from .compression import ModelInfo, _pick_chunk_bounds, _pick_archive_chunk, summarize_chunk
from .ids import combined_hash, message_hash
from .store import Store

logger = logging.getLogger(__name__)


_CHARS_PER_TOKEN = 4


class SpeculativeWorker:
    """Single-threaded background summarizer.

    Owned by the ContextEngine — one instance per session. Safe to call
    ``schedule()`` from the main thread at any time; duplicate schedule
    calls for the same chunk are deduplicated by source_hash.
    """

    def __init__(
        self, store: Store, cfg: Dict[str, Any], model_info: ModelInfo,
        *, engine: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._model_info = model_info
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending_source_hash: Optional[str] = None
        self._enabled = bool(cfg.get("compression", {}).get("speculative", True))
        self._shutdown = False
        # Optional back-reference to the engine so we can check
        # ``_compression_halted`` in schedule(). When the main path
        # halts, speculation stays quiet — otherwise it could land a
        # successful summary while the system is supposedly halted,
        # producing an inconsistent picture for operators.
        self._engine = engine

    # -- public API ------------------------------------------------------

    def schedule(self, messages: List[Dict[str, Any]], *, protect_first_n: int = 3) -> None:
        """Kick off speculative summarization of the prospective chunk.

        Cheap and non-blocking: picks the chunk, computes its
        source_hash, and either discovers an existing L1 (skipping) or
        dispatches a background summarization.

        Respects the engine's halt flag — if main compression has
        halted (e.g., because the auxiliary summarizer is failing),
        speculation stays out of the way too. Otherwise we'd race in
        a successful summary against the halt and produce inconsistent
        signal about whether the system is actually broken.
        """
        if not self._enabled or self._shutdown:
            return
        if getattr(self, "_engine", None) is not None:
            if getattr(self._engine, "_compression_halted", False):
                return

        # Reconcile any new content from ``messages`` into the archive
        # before walking. In production this is a no-op (engine has
        # already archived via on_turn_end / compress). For tests that
        # call schedule() directly with a fresh messages list, this
        # ensures the archive has the content speculation will walk.
        if self._engine is not None and hasattr(self._engine, "_reconcile_with_archive"):
            try:
                self._engine._reconcile_with_archive(messages)
            except Exception as e:
                logger.debug("speculative: reconcile failed: %s", e)
        else:
            self._reconcile_messages(messages)

        # Walk the archive's uncovered messages — same source the main
        # compress() path uses. If we walked the live ``messages`` list
        # instead, our chunk would differ from main's chunk (markers vs
        # raw, position drift), source_hashes wouldn't match, and the
        # archive-cache hit on the main path would miss.
        bounds = _chunk_bounds_from_archive(
            self._store, self._cfg, protect_first_n,
        )
        if bounds is None:
            return
        chunk_msgs, chunk_hashes = bounds
        source_hash = combined_hash(chunk_hashes)

        # Already summarized? Archive is the cache.
        if self._store.find_summary_by_source_hash(source_hash, level=1) is not None:
            return

        with self._lock:
            if self._pending_source_hash == source_hash:
                return
            if self._thread is not None and self._thread.is_alive():
                return

            snapshot = Snapshot(chunk_msgs, 0, len(chunk_msgs))
            self._pending_source_hash = source_hash
            t = threading.Thread(
                target=self._run,
                args=(source_hash, snapshot),
                name=f"autobio-spec-{source_hash[:6]}",
                daemon=True,
            )
            self._thread = t
            t.start()
            logger.debug("autobio: speculative worker started for %s", source_hash)

    # How long to wait for an in-flight summarization on session end.
    # If the LLM call returns within this window the worker can write
    # its L1 to the archive before the store closes; if not, the write
    # would land on a closed connection and the summary would be lost.
    _SHUTDOWN_JOIN_TIMEOUT = 30.0

    def _reconcile_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Lightweight reconcile when no engine is wired (rare — only
        in unit tests that instantiate SpeculativeWorker directly).
        Mirrors engine._reconcile_with_archive minimally.
        """
        from .render import is_recall_message
        with self._store._lock:
            row = self._store._conn.execute(
                "SELECT COALESCE(MAX(position), -1) FROM messages "
                "WHERE session_id = ?",
                (self._store.session_id,),
            ).fetchone()
        next_pos = (row[0] if row else -1) + 1
        for msg in messages:
            if is_recall_message(msg):
                continue
            try:
                h = message_hash(msg)
                if self._store.get_message(h) is None:
                    self._store.upsert_message(h, msg, position=next_pos)
                    next_pos += 1
            except Exception:
                pass

    def shutdown(self) -> None:
        self._shutdown = True
        thread = None
        with self._lock:
            thread = self._thread
            pending = self._pending_source_hash
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._SHUTDOWN_JOIN_TIMEOUT)
            if thread.is_alive():
                logger.error(
                    "autobio speculative: shutdown timed out after %.1fs with "
                    "worker still running (thread=%s pending_source_hash=%s). "
                    "Any L1 the worker tries to commit after this point will "
                    "hit a closed database and be lost.",
                    self._SHUTDOWN_JOIN_TIMEOUT, thread.name, pending,
                )

    # -- worker body -----------------------------------------------------

    def _run(self, source_hash: str, snapshot: "Snapshot") -> None:
        from .compression import SummarizationFailed
        try:
            summary_id = summarize_chunk(
                snapshot.messages,
                snapshot.chunk_start,
                snapshot.chunk_end,
                self._store,
                self._cfg,
                self._model_info,
            )
            if summary_id is not None:
                logger.info("autobio speculative: L1 %s ready in archive", summary_id)
        except SummarizationFailed as e:
            # Speculative failure does NOT halt the engine — the main
            # path will hit the same failure when compress() fires and
            # will set the halt flag from there. We just log so the
            # pattern is visible if the user checks logs.
            logger.error(
                "autobio speculative: summarizer returned empty for %s — "
                "main path will halt on its own attempt: %s",
                source_hash, e,
            )
        except Exception as e:
            logger.warning("autobio speculative worker error: %s", e)
        finally:
            with self._lock:
                self._pending_source_hash = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Snapshot:
    __slots__ = ("messages", "chunk_start", "chunk_end")

    def __init__(self, messages: List[Dict[str, Any]], chunk_start: int, chunk_end: int):
        self.messages = messages
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end


def _snapshot(messages: List[Dict[str, Any]], bounds: "tuple[int, int]") -> Snapshot:
    return Snapshot([dict(m) for m in messages], bounds[0], bounds[1])


def _chunk_bounds_from_archive(
    store: Store,
    cfg: Dict[str, Any],
    protect_first_n: int,
) -> Optional["tuple[List[Dict[str, Any]], List[str]]"]:
    """Pick the prospective chunk for speculation, archive-driven.

    Returns ``(chunk_messages, chunk_hashes)`` or None if no chunk is
    eligible. Calls the same archive-walking picker the main compress
    path uses, so source_hashes match across the two paths and the
    archive-cache hit on main works as designed.
    """
    from .compression import _materialize_chunk

    compression_cfg = cfg.get("compression", {})
    chunk_tokens = int(compression_cfg.get("chunk_tokens", 10_000))
    protect_last = chunk_tokens * 2
    weights_cfg = compression_cfg.get("weights")

    chunk_hashes, _doc_context = _pick_archive_chunk(
        store,
        chunk_token_target=chunk_tokens,
        protect_last_tokens=protect_last,
        protect_first_n=protect_first_n,
        weights_cfg=weights_cfg,
    )
    if not chunk_hashes:
        return None
    chunk_msgs = _materialize_chunk(store, chunk_hashes)
    return (chunk_msgs, chunk_hashes)


def _chunk_bounds_for(
    messages: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    protect_first_n: int,
    *,
    store: Optional[Store] = None,
) -> Optional["tuple[int, int]"]:
    """Pick the prospective chunk for speculation.

    When a store is provided, consult protected ranges so speculation
    doesn't waste LLM calls summarizing pinned content (which compress
    would skip anyway at fold time). With no store, falls back to the
    unconstrained picker — used in tests.
    """
    compression_cfg = cfg.get("compression", {})
    chunk_tokens = int(compression_cfg.get("chunk_tokens", 10_000))
    protect_last = chunk_tokens * 2
    weights_cfg = compression_cfg.get("weights")
    protected_ranges = (
        store.find_ranges_in_view([message_hash(m) for m in messages])
        if store is not None else []
    )
    start, end = _pick_chunk_bounds(
        messages, protect_first_n, chunk_tokens, protect_last,
        protected_ranges=protected_ranges,
        weights_cfg=weights_cfg,
    )
    if end <= start:
        return None
    return (start, end)

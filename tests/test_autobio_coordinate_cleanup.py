"""Tests for the coordinate-cleanup refactor.

Exercises the four structural changes:

1. **Picker walks archive directly.** A mid-archive coverage gap must
   not produce a chunk that spans it.
2. **Markers carry an explicit ``_autobio_marker`` flag.** The flag
   round-trips through render output and ``is_marker_message``
   recognizes it without falling back to the prefix heuristic.
3. **Tool-call shape canonical end-to-end.** Stored rows are canonical
   OpenAI form; render passes them through without re-inflation.
4. **Merge requires position contiguity.** A pin-induced gap between
   L1s defers a non-adjacent merge until contiguity exists.

Each scenario in ``coordinate-cleanup-refactor.md`` corresponds to one
test below. When the harness finds a regression in this area, add a
reduced repro here so the regression cannot recur silently.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resp(text: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    return r


# ---------------------------------------------------------------------------
# Change 1: picker walks archive directly
# ---------------------------------------------------------------------------

class PickerArchiveWalk(unittest.TestCase):
    """The picker must respect archive position. A gap in coverage
    (some middle messages already folded into a different summary)
    must NOT produce a chunk that spans the gap.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_pwa_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_picker_walks_archive_skipping_covered(self) -> None:
        """Synthetic archive: positions 0..29 with positions 10..14
        already covered by L1-fake. The picker should only consider
        the post-gap run (positions 15..29) — its chunk hashes must be
        position-contiguous in 15..29 and never include any hash from
        0..14.
        """
        from plugins.autobio.compression import _pick_archive_chunk
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "pwa")

        # Insert 30 messages with explicit positions.
        msgs: List[Dict[str, Any]] = []
        for i in range(30):
            role = "user" if i % 2 == 0 else "assistant"
            msg = {"role": role, "content": f"msg{i} " + ("x" * 800)}
            msgs.append(msg)
            store.upsert_message(message_hash(msg), msg, turn=i, position=i)

        # Insert a fake L1 covering positions 0..14.
        early_hashes = [message_hash(m) for m in msgs[:15]]
        fake_l1 = store.insert_summary(
            level=1, source_hashes=early_hashes,
            source_hash="fake-early", content="early summary",
            tokens=100, compression_generations=1,
        )
        store.mark_compressed(early_hashes, fake_l1)

        # Now pick with low chunk_token_target — the available run is
        # 15..29 only; the picker should never reach into 0..14.
        chunk_hashes, _doc = _pick_archive_chunk(
            store,
            chunk_token_target=2_000,
            protect_last_tokens=0,
            protect_first_n=1,
        )

        self.assertGreater(len(chunk_hashes), 0,
                           "picker should find an eligible chunk in 15..29")
        early_set = set(early_hashes)
        for h in chunk_hashes:
            self.assertNotIn(
                h, early_set,
                f"picker leaked into covered region: hash={h}",
            )

        # Verify positional contiguity within the chunk.
        positions = []
        for h in chunk_hashes:
            row = store.get_message(h)
            self.assertIsNotNone(row)
            positions.append(int(row["position"]))
        for a, b in zip(positions, positions[1:]):
            self.assertEqual(
                a + 1, b,
                f"non-contiguous positions in chunk: {a} → {b}",
            )

        store.close()


# ---------------------------------------------------------------------------
# Change 2: marker flag
# ---------------------------------------------------------------------------

class MarkerFlag(unittest.TestCase):
    """Markers carry ``_autobio_marker = True`` and are recognized
    without the prefix fallback firing."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_mf_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_marker_flag_survives_render_round_trip(self) -> None:
        """End-to-end: trigger a fold, get the rendered list, verify
        the marker has the flag, then re-feed the list through
        ``_reconcile_with_archive`` and verify the marker is skipped
        (no warning fired about prefix-only recognition).
        """
        import logging

        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.render import _MARKER_FLAG, is_marker_message

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("mf", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300
        eng.cfg["compression"]["speculative"] = False
        if eng._speculative is not None:
            eng._speculative._enabled = False

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(30):
            msgs.append({"role": "user",
                         "content": f"u{i} " + ("x" * 2500)})
            msgs.append({"role": "assistant",
                         "content": f"a{i} " + ("y" * 2500)})

        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("memory")):
            rendered = eng.compress(msgs)

        # Find the marker(s) in the rendered output.
        markers = [m for m in rendered if is_marker_message(m)]
        self.assertGreater(
            len(markers), 0, "compress() should have produced a marker",
        )
        for m in markers:
            self.assertIs(
                m.get(_MARKER_FLAG), True,
                f"marker missing {_MARKER_FLAG} flag: {m}",
            )

        # Re-feed the rendered list. The reconcile path should skip
        # markers without firing the prefix-fallback warning.
        autobio_logger = logging.getLogger("plugins.autobio.render")
        captured: List[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        cap = _Capture(level=logging.WARNING)
        autobio_logger.addHandler(cap)
        try:
            eng._reconcile_with_archive(rendered)
        finally:
            autobio_logger.removeHandler(cap)

        prefix_fallback = [
            r for r in captured
            if "marker recognized by prefix only" in r.getMessage()
        ]
        self.assertEqual(
            prefix_fallback, [],
            "prefix-fallback warning fired during round-trip "
            "(flag should have been the primary signal)",
        )

        eng.on_session_end("mf", [])

    def test_prefix_fallback_fires_on_flagless_marker(self) -> None:
        """If a caller hand-builds a marker without the flag (or the
        flag was stripped during a JSON round-trip), the prefix
        fallback should still recognize it AND log a warning so the
        gap is investigable.
        """
        import logging

        from plugins.autobio.render import is_marker_message

        flagless = {
            "role": "user",
            "content": "[autobio:memory L1-0001 · L1 · 200t · gen 1 · "
                       "call cm_manage_context to expand]",
        }

        autobio_logger = logging.getLogger("plugins.autobio.render")
        captured: List[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        cap = _Capture(level=logging.WARNING)
        autobio_logger.addHandler(cap)
        try:
            recognized = is_marker_message(flagless)
        finally:
            autobio_logger.removeHandler(cap)

        self.assertTrue(recognized,
                        "prefix fallback should still recognize the marker")
        self.assertTrue(
            any("prefix only" in r.getMessage() for r in captured),
            "expected a warning about prefix-only recognition",
        )


# ---------------------------------------------------------------------------
# Change 3: canonical tool-call shape
# ---------------------------------------------------------------------------

class CanonicalToolCallShape(unittest.TestCase):
    """The store now persists the canonical OpenAI tool-call shape
    verbatim. Round-tripping a message with tool_calls through the
    archive produces a shape-identical dict.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_tc_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_canonical_form_round_trips_unchanged(self) -> None:
        from plugins.autobio.ids import message_hash
        from plugins.autobio.render import _msg_to_openai
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "tc")
        canonical_msg = {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{
                "id": "call-abc",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": '{"command": "ls"}',
                },
            }],
        }
        h = message_hash(canonical_msg)
        store.upsert_message(h, canonical_msg, turn=0, position=0)
        row = store.get_message(h)

        # Stored JSON should be canonical.
        stored = json.loads(row["tool_calls"])
        self.assertEqual(len(stored), 1)
        self.assertIsInstance(stored[0].get("function"), dict)
        self.assertEqual(stored[0]["function"]["name"], "terminal")

        # Render output is also canonical (no re-inflation step).
        rendered = _msg_to_openai(row)
        self.assertEqual(rendered["tool_calls"][0]["type"], "function")
        self.assertEqual(
            rendered["tool_calls"][0]["function"]["name"], "terminal",
        )
        self.assertEqual(
            rendered["tool_calls"][0]["function"]["arguments"],
            '{"command": "ls"}',
        )

        store.close()

    def test_legacy_flat_form_migrated_on_open(self) -> None:
        """A store opened with pre-canonical (flat) tool_calls rows
        should auto-migrate them to canonical shape during
        ``_run_migrations``."""
        from plugins.autobio.store import close_store, get_store

        # First open: write canonical via upsert_message (we'll
        # overwrite with flat directly to simulate legacy data).
        store = get_store(self.tmp, "legacy")
        canonical_msg = {
            "role": "assistant",
            "content": "x",
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "fn", "arguments": "{}"},
            }],
        }
        store.upsert_message("legacy-hash", canonical_msg, turn=0, position=0)

        # Stomp the row with flat-form JSON to simulate legacy data.
        flat_json = json.dumps(
            [{"id": "call-1", "name": "fn", "arguments": "{}"}],
        )
        with store._lock:
            store._conn.execute(
                "UPDATE messages SET tool_calls = ? WHERE hash = ?",
                (flat_json, "legacy-hash"),
            )
        store.close()
        close_store(self.tmp, "legacy")

        # Reopen — migration should run.
        store2 = get_store(self.tmp, "legacy")
        row = store2.get_message("legacy-hash")
        migrated = json.loads(row["tool_calls"])
        self.assertEqual(len(migrated), 1)
        self.assertIsInstance(
            migrated[0].get("function"), dict,
            "flat-form row should have been re-inflated to canonical",
        )
        self.assertEqual(migrated[0]["function"]["name"], "fn")
        store2.close()


# ---------------------------------------------------------------------------
# Change 4: merge contiguity
# ---------------------------------------------------------------------------

class MergeContiguity(unittest.TestCase):
    """Two L1s only merge if their position coverage is adjacent. A
    pin-induced gap between L1-A (positions 0..4) and L1-C (positions
    10..14), with L1-B's worth of pinned messages in between but no
    L1-B summary, should NOT produce an L2 covering A+C — that would
    render as two separate markers.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_mc_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_merge_contiguity_defers_non_adjacent(self) -> None:
        from plugins.autobio.compression import (
            ModelInfo, _find_contiguous_mergeable_group,
            merge_if_ready,
        )
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "mc")

        # Insert 15 messages with positions 0..14.
        msgs: List[Dict[str, Any]] = []
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"m{i}"}
            msgs.append(m)
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        # L1-A covers positions 0..4 (group of 5).
        a_hashes = [message_hash(m) for m in msgs[:5]]
        l1a = store.insert_summary(
            level=1, source_hashes=a_hashes, source_hash="sa",
            content="a", tokens=50, compression_generations=1,
        )
        store.mark_compressed(a_hashes, l1a)

        # L1-B covers positions 5..9 (group of 5).
        b_hashes = [message_hash(m) for m in msgs[5:10]]
        l1b = store.insert_summary(
            level=1, source_hashes=b_hashes, source_hash="sb",
            content="b", tokens=50, compression_generations=1,
        )
        store.mark_compressed(b_hashes, l1b)

        # Skip 10..11 — these stay uncovered (a "pin gap").
        # L1-C covers positions 12..14 (group of 3).
        c_hashes = [message_hash(m) for m in msgs[12:15]]
        l1c = store.insert_summary(
            level=1, source_hashes=c_hashes, source_hash="sc",
            content="c", tokens=50, compression_generations=1,
        )
        store.mark_compressed(c_hashes, l1c)

        # With threshold=2, the contiguous group {L1-A, L1-B} (covering
        # 0..9) is mergeable; {L1-C} alone is not (size 1). The picker
        # must NOT span the gap between L1-B (max pos 9) and L1-C
        # (min pos 12).
        group = _find_contiguous_mergeable_group(store, level=1, threshold=2)
        self.assertIsNotNone(group)
        ids = sorted(s["id"] for s in group)
        self.assertEqual(
            ids, sorted([l1a, l1b]),
            f"contiguous group should be A+B only; got {ids}",
        )

        # End-to-end: merge_if_ready merges only the contiguous pair.
        cfg = {"compression": {
            "merge_threshold": 2, "max_levels": 3,
            "summary_target_tokens": 100,
        }}
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("merged")):
            created = merge_if_ready(store, cfg, ModelInfo(model="t"))

        self.assertEqual(
            len(created), 1,
            f"only A+B should merge; got {len(created)} merges: {created}",
        )

        # L1-A and L1-B are now merged_into the new L2; L1-C is still
        # active and unmerged.
        self.assertIsNotNone(store.get_summary(l1a)["merged_into"])
        self.assertIsNotNone(store.get_summary(l1b)["merged_into"])
        self.assertIsNone(store.get_summary(l1c)["merged_into"])

        store.close()


if __name__ == "__main__":
    unittest.main()

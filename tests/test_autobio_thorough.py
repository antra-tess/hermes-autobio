"""Thorough mock-based tests for the autobio plugin.

Targets invariants and failure modes the e2e suite only brushes past:

- Archive round-trip integrity (compress → expand → content preserved)
- Rollback safety (speculative cache naturally invalidates on /undo)
- Full L1 → L2 → L3 provenance chain
- Tool-call/result orphan sanitization after compression
- Per-tool retention actually differs with per-tool config
- Store survives close/reopen (FTS5 on-disk persistence)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
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


class ArchiveRoundTrip(unittest.TestCase):
    """Compress → mark compressed → expand → original content recovered."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_rt_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_l1_expand_returns_original_messages(self) -> None:
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.search import expand_summary

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("rt", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300

        # Distinctive content we can identify post-expansion.
        originals: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "User says UNIQUE_MARKER_ALPHA"},
            {"role": "assistant", "content": "Assistant says UNIQUE_MARKER_BETA"},
        ]
        for i in range(30):
            originals.append({"role": "user", "content": f"turn-{i}-user " + ("x" * 2000)})
            originals.append({"role": "assistant", "content": f"turn-{i}-asst " + ("y" * 2000)})
        originals.append({"role": "user", "content": "Final user message"})

        with patch("agent.auxiliary_client.call_llm", return_value=_resp("memory summary")):
            eng.compress(originals)

        l1 = eng._store.list_summaries(1, include_merged=False)
        self.assertEqual(len(l1), 1)

        expanded = expand_summary(eng._store, l1[0]["id"])
        # Expanded content must contain some of the original message bodies.
        expanded_text = "\n".join(str(m.get("content") or "") for m in expanded)
        self.assertIn("turn-0-user", expanded_text, "original content lost in L1 expansion")
        self.assertIn("turn-0-asst", expanded_text)

        eng.on_session_end("rt", originals)


class RollbackSafety(unittest.TestCase):
    """Rollback semantics under archive-first (post-refactor).

    Key invariant: once a chunk has been summarized, its L1 is an
    authoritative memory. A /undo that removes messages *after* the
    chunk does not invalidate the summary — the memory is still valid.
    A /undo that reaches back into the chunk produces a different
    source_hash, so a fresh summary is created; the original L1
    remains in the archive (harmless; it is a memory of a chunk that
    no longer appears in the live view).
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_rb_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_engine_and_messages(self):
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("rb", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(40):
            msgs.append({"role": "user", "content": f"u{i} " + ("a" * 3000)})
            msgs.append({"role": "assistant", "content": f"a{i} " + ("b" * 3000)})
        return eng, msgs

    def test_tail_rollback_preserves_cached_summary(self) -> None:
        """Dropping messages AFTER the chunk must not force re-summarization."""
        eng, msgs = self._make_engine_and_messages()
        calls = [0]
        def fake(**k):
            calls[0] += 1
            return _resp("speculative text")

        with patch("agent.auxiliary_client.call_llm", side_effect=fake):
            eng._speculative.schedule(msgs, protect_first_n=3)
            while eng._speculative._thread and eng._speculative._thread.is_alive():
                time.sleep(0.02)
            self.assertEqual(calls[0], 1, "speculative worker should have called LLM once")

            # /undo removes the last 4 messages — all in the tail,
            # nowhere near the chunk at the front.
            rolled = msgs[:-4]
            eng.compress(rolled)
            self.assertEqual(calls[0], 1, "LLM should NOT be called again — chunk unchanged")

        # Exactly one L1 in the archive, reused by compress().
        l1_total = eng._store.list_summaries(1, include_merged=True)
        self.assertEqual(len(l1_total), 1)
        eng.on_session_end("rb", msgs)

    def test_chunk_rollback_under_projection_model(self) -> None:
        """Under the projection model the archive is append-only. A
        runner-side "rollback" (passing a modified messages list to
        compress) just appends the new content; the original chunk
        content remains in the archive and is folded the same way.
        The original L1's source_hash still matches, so no new LLM
        call is made, and the rolled-back message lives uncovered in
        the tail until a future fold picks it up.

        This differs from the pre-projection invariant (rollback into
        the chunk → fresh L1) — the new model preserves provenance
        more strictly: nothing the runner says can erase what was
        archived.
        """
        eng, msgs = self._make_engine_and_messages()
        calls = [0]
        def fake(**k):
            calls[0] += 1
            return _resp(f"memory {calls[0]}")

        with patch("agent.auxiliary_client.call_llm", side_effect=fake):
            eng._speculative.schedule(msgs, protect_first_n=3)
            while eng._speculative._thread and eng._speculative._thread.is_alive():
                time.sleep(0.02)

            rolled = msgs[:4] + [
                {"role": "user", "content": "rewritten early message " + ("z" * 3000)}
            ] + msgs[5:]
            eng.compress(rolled)
            # Same source_hash → same L1, no second LLM call.
            self.assertEqual(calls[0], 1,
                             "speculative L1 reused; archive append-only")

        # Exactly one L1 in archive; the rolled-in message is appended
        # uncovered, awaiting a future fold.
        l1_total = eng._store.list_summaries(1, include_merged=True)
        self.assertEqual(len(l1_total), 1)
        eng.on_session_end("rb", msgs)


class MultiLevelMerge(unittest.TestCase):
    """Six L1s → one L2 → (with six L2s) → one L3, chain preserved."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_ml_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_three_level_chain(self) -> None:
        from plugins.autobio.compression import ModelInfo, merge_if_ready
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "ml")
        mi = ModelInfo(model="t")
        cfg = {"compression": {"merge_threshold": 3, "max_levels": 3, "summary_target_tokens": 200}}

        # Seed 9 L1s so that merging at threshold=3 yields 3 L2s, which
        # then merge into 1 L3. Also insert content so FTS5 indexing
        # exercises all paths.
        for i in range(9):
            store.insert_summary(
                level=1, source_hashes=[f"h{i}"], source_hash=f"h{i}",
                content=f"Memory #{i}: specific detail {i}",
                tokens=100, compression_generations=1,
                first_turn=i * 10, last_turn=i * 10 + 9,
            )

        call_counter = [0]
        def counting(**kwargs):
            call_counter[0] += 1
            return _resp(f"merged memory batch {call_counter[0]}")

        with patch("agent.auxiliary_client.call_llm", side_effect=counting):
            created = merge_if_ready(store, cfg, mi)

        # Expect: 3 L2s created from L1 batches, then 1 L3 from the 3 L2s.
        self.assertEqual(len(created), 4, f"expected 4 created summaries, got {created}")

        l2_active = store.list_summaries(2, include_merged=False)
        l3_active = store.list_summaries(3, include_merged=False)
        self.assertEqual(len(l2_active), 0, "L2s should be merged into L3")
        self.assertEqual(len(l3_active), 1, "expected exactly 1 L3")

        # Verify the chain: L3's source_hashes are L2 IDs, each L2's
        # source_hashes are 3 L1 IDs.
        l3 = l3_active[0]
        l3_children = json.loads(l3["source_hashes"])
        self.assertEqual(len(l3_children), 3)
        for l2_id in l3_children:
            l2 = store.get_summary(l2_id)
            self.assertEqual(l2["merged_into"], l3["id"])
            l2_children = json.loads(l2["source_hashes"])
            self.assertEqual(len(l2_children), 3)
            for l1_id in l2_children:
                l1 = store.get_summary(l1_id)
                self.assertEqual(l1["merged_into"], l2_id)
                self.assertEqual(l1["level"], 1)

        self.assertEqual(l3["compression_generations"], 3)
        store.close()


class OrphanToolPairs(unittest.TestCase):
    """Chunk that splits an assistant+tool_result group produces valid output."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_op_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_dangling_tool_ids_after_compression(self) -> None:
        from plugins.autobio.compression import ModelInfo, compress_to_l1
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "op")
        mi = ModelInfo(model="t")

        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"u{i} " + ("x" * 1500)})
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "t", "arguments": json.dumps({"i": i})},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result{i} " + ("y" * 1500)})
            msgs.append({"role": "assistant", "content": f"follow-up {i}"})

        cfg = {"compression": {"chunk_tokens": 3000, "summary_target_tokens": 300}}
        with patch("agent.auxiliary_client.call_llm", return_value=_resp("compacted")):
            result = compress_to_l1(
                msgs, store, cfg, mi,
                protect_first_n=3, protect_last_tokens=5000,
            )

        # Every surviving tool result must have a matching assistant tool_call.
        call_ids = set()
        for m in result.new_messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        call_ids.add(cid)
        for m in result.new_messages:
            if m.get("role") == "tool":
                cid = m.get("tool_call_id", "")
                self.assertIn(
                    cid, call_ids,
                    f"orphan tool result {cid} in post-compression list",
                )

        # Every surviving tool_call must have a matching result.
        result_ids = {m["tool_call_id"] for m in result.new_messages if m.get("role") == "tool"}
        for m in result.new_messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        self.assertIn(
                            cid, result_ids,
                            f"tool_call {cid} has no matching result",
                        )
        store.close()


class PerToolRetention(unittest.TestCase):
    """Per-tool retention config produces different prune behavior."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="autobio_pt_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tight_retention_summarizes_older_terminal_results(self) -> None:
        from plugins.autobio.store import get_store
        from plugins.autobio.tool_pruning import prune_tool_results

        store = get_store(self.tmp, "pt")
        msgs: List[Dict[str, Any]] = []
        # Three terminal calls with DIFFERENT big outputs so dedup
        # doesn't collapse them before the summarize pass runs.
        for i in range(3):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": f"t{i}", "type": "function",
                "function": {"name": "terminal", "arguments": json.dumps({"command": f"ls {i}"})},
            }]})
            # Distinctive content per-call so content hashes differ.
            big_i = f"CALL_{i}_OUTPUT\n" + ("line\n" * 500)
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": big_i})

        # Tight retention (700 tokens) — only the NEWEST should stay full.
        tight_cfg = {
            "default": {"full_retention_tokens": 30_000, "summary_min_chars": 200, "argument_truncate_chars": 500},
            "per_tool": {"terminal": {"full_retention_tokens": 700}},
        }
        _, stats = prune_tool_results([m.copy() for m in msgs], store, tight_cfg)
        self.assertEqual(stats.per_tool.get("terminal", {}).get("summarized", 0), 2,
                         f"expected 2 terminal summarized; got {stats.per_tool}")
        self.assertEqual(stats.per_tool.get("terminal", {}).get("full", 0), 1)

        # Generous retention — all three stay full.
        loose_cfg = {
            "default": {"full_retention_tokens": 30_000, "summary_min_chars": 200, "argument_truncate_chars": 500},
            "per_tool": {"terminal": {"full_retention_tokens": 10_000}},
        }
        # Fresh copy (prune is non-destructive but let's be safe).
        _, stats = prune_tool_results([m.copy() for m in msgs], store, loose_cfg)
        self.assertEqual(stats.per_tool.get("terminal", {}).get("summarized", 0), 0)
        self.assertEqual(stats.per_tool.get("terminal", {}).get("full", 0), 3)
        store.close()


class StorePersistence(unittest.TestCase):
    """Data survives store close+reopen."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="autobio_sp_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summaries_and_fts_persist_across_reopen(self) -> None:
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "sp")
        store.insert_summary(
            level=1, source_hashes=["h"], source_hash="h",
            content="The rate_limiter config is in config.py line 42.",
            tokens=50, compression_generations=1,
        )
        store.upsert_message(
            "deadbeef00001111",
            {"role": "user", "content": "What about rate_limiter.max?"},
            turn=5, position=12,
        )
        close_store(self.tmp, "sp")

        store2 = get_store(self.tmp, "sp")
        # Summary + message survived
        l1 = store2.list_summaries(1, include_merged=False)
        self.assertEqual(len(l1), 1)
        self.assertIn("rate_limiter", l1[0]["content"])
        # Still findable via FTS5
        hits = store2.fts_summaries("rate_limiter")
        self.assertGreaterEqual(len(hits), 1)
        msg_hits = store2.fts_messages("rate_limiter")
        self.assertGreaterEqual(len(msg_hits), 1)
        close_store(self.tmp, "sp")


class SummarizeFoldSemantics(unittest.TestCase):
    """New primitives introduced by the summarize/fold refactor."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_sf_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _messages(self):
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(30):
            msgs.append({"role": "user", "content": f"unique-user-{i} " + ("x" * 2500)})
            msgs.append({"role": "assistant", "content": f"unique-asst-{i} " + ("y" * 2500)})
        return msgs

    def test_summarize_chunk_is_idempotent_by_source_hash(self) -> None:
        from plugins.autobio.compression import ModelInfo, summarize_chunk, _pick_chunk_bounds
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "idem")
        mi = ModelInfo(model="t")
        cfg = {"compression": {"summary_target_tokens": 300}}

        msgs = self._messages()
        start, end = _pick_chunk_bounds(msgs, 3, 4_000, 8_000)
        self.assertLess(start, end)

        calls = [0]
        def fake(**k):
            calls[0] += 1
            return _resp("memory")

        with patch("agent.auxiliary_client.call_llm", side_effect=fake):
            id1 = summarize_chunk(msgs, start, end, store, cfg, mi, first_turn=0, last_turn=10)
            id2 = summarize_chunk(msgs, start, end, store, cfg, mi, first_turn=0, last_turn=10)

        self.assertIsNotNone(id1)
        self.assertEqual(id1, id2, "second call must return existing L1")
        self.assertEqual(calls[0], 1, "LLM must be called only once")
        store.close()

    def test_fold_then_unfold_recovers_original_content(self) -> None:
        from plugins.autobio.compression import _pick_chunk_bounds, unfold
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("fu", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300
        eng.cfg["compression"]["speculative"] = False
        if eng._speculative is not None:
            eng._speculative._enabled = False

        msgs = self._messages()
        start, end = _pick_chunk_bounds(msgs, 3, 4_000, 8_000)
        original_chunk_contents = [
            m.get("content", "") for m in msgs[start:end]
        ]

        with patch("agent.auxiliary_client.call_llm", return_value=_resp("folded memory")):
            folded = eng.compress(msgs)

        # Exactly one L1 was written.
        l1s = eng._store.list_summaries(1, include_merged=False)
        self.assertEqual(len(l1s), 1)
        summary_id = l1s[0]["id"]
        self.assertLess(len(folded), len(msgs), "fold should shrink the list")

        restored = unfold(folded, summary_id, eng._store)
        restored_text = "\n".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in restored
        )
        # Every distinctive user/assistant content from the chunk should
        # reappear after unfold.
        for original in original_chunk_contents:
            if isinstance(original, str) and "unique-" in original:
                self.assertIn(
                    original[:40], restored_text,
                    f"unfold lost content: {original[:40]!r}",
                )
        eng.on_session_end("fu", [])

    def test_fold_requires_existing_summary(self) -> None:
        from plugins.autobio.compression import fold_chunk
        from plugins.autobio.store import get_store

        store = get_store(self.tmp, "req")
        msgs = self._messages()
        with self.assertRaises(ValueError):
            fold_chunk(msgs, 3, 10, "L1-9999", store)
        store.close()


class HashIdempotency(unittest.TestCase):
    """Message hashing is stable and order-dependent where it should be."""

    def test_hash_stability_across_reorderings(self) -> None:
        from plugins.autobio.ids import combined_hash, message_hash

        m = {"role": "user", "content": "hello"}
        h1 = message_hash(m)
        h2 = message_hash(m)
        self.assertEqual(h1, h2)

        # Content change → different hash
        m2 = {"role": "user", "content": "hello!"}
        self.assertNotEqual(message_hash(m2), h1)

        # Role change → different hash
        m3 = {"role": "assistant", "content": "hello"}
        self.assertNotEqual(message_hash(m3), h1)

        # tool_calls participation
        asst_a = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
        ]}
        asst_b = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path":"b.py"}'}},
        ]}
        self.assertNotEqual(message_hash(asst_a), message_hash(asst_b))

        # combined_hash order matters
        self.assertNotEqual(
            combined_hash(["a", "b", "c"]),
            combined_hash(["c", "b", "a"]),
        )


class Documents(unittest.TestCase):
    """Document (kind=doc_*) pin behavior and the auto-detect pipeline."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_doc_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_engine(self):
        from plugins.autobio.engine import AutobioContextEngine
        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("doc", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300
        eng.cfg["compression"]["speculative"] = False
        if eng._speculative is not None:
            eng._speculative._enabled = False
        return eng

    def _big_conversation(self, n_turns: int = 40, size: int = 3000):
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(n_turns):
            msgs.append({"role": "user", "content": f"u{i}-{i} " + ("A" * size)})
            msgs.append({"role": "assistant", "content": f"a{i}-{i} " + ("B" * size)})
        return msgs

    def test_plain_pin_is_never_folded(self) -> None:
        """A kind='pin' range is skipped even when docs would be eligible."""
        from plugins.autobio.compression import _pick_chunk_bounds
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()
        msgs = self._big_conversation(20, size=4000)
        # Archive everything first.
        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        # Pin the first 10 messages.
        start_h = message_hash(msgs[3])
        end_h = message_hash(msgs[12])
        eng._store.add_pin(start_h, end_h, reason="keep", kind="pin")

        ranges = eng._store.find_ranges_in_view([message_hash(m) for m in msgs])
        start, end = _pick_chunk_bounds(
            msgs, 3, 4_000, 8_000,
            protected_ranges=ranges,
            is_eligible=lambda r: True,   # force max permissiveness
        )
        self.assertTrue(start > 12 or end == start,
                        f"chunk {start}..{end} overlaps the plain pin")
        eng.on_session_end("doc", [])

    def test_doc_skipped_at_low_pressure(self) -> None:
        """When non-doc content is available, a doc is never folded."""
        from plugins.autobio.compression import _pick_chunk_bounds
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()
        msgs = self._big_conversation(20, size=4000)
        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        # Mark messages 3..8 as a doc.
        start_h = message_hash(msgs[3])
        end_h = message_hash(msgs[8])
        eng._store.add_pin(
            start_h, end_h, reason="log", kind="doc_chronological",
            meta={"title": "some log"},
        )

        # Size-pressure policy: first pass forbids docs.
        ranges = eng._store.find_ranges_in_view([message_hash(m) for m in msgs])
        start, end = _pick_chunk_bounds(
            msgs, 3, 4_000, 8_000,
            protected_ranges=ranges,
            is_eligible=lambda r: False,    # non-doc pass
        )
        # Chunk should start after the doc's end_index (8), not inside it.
        self.assertTrue(start > 8, f"doc was folded at low pressure: chunk {start}..{end}")
        eng.on_session_end("doc", [])

    def test_chronological_doc_folds_top_first_under_pressure(self) -> None:
        """Under eligibility, a chronological doc folds from its top,
        the pin's start_hash advances, the archive retains the rest."""
        from plugins.autobio.compression import ModelInfo, compress_step
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()
        msgs = self._big_conversation(10, size=4000)
        # Seed archive so find_ranges_in_view can resolve positions.
        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        # Mark a 6-message doc starting at index 3.
        start_h = message_hash(msgs[3])
        end_h = message_hash(msgs[8])
        pin_id = eng._store.add_pin(
            start_h, end_h, reason="log", kind="doc_chronological",
            meta={"title": "big log", "original_tokens": 30_000},
        )

        # Force doc eligibility explicitly — this bypasses size-pressure policy.
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("doc chunk memory")):
            result = compress_step(
                msgs, eng._store, eng.cfg,
                ModelInfo(model="t"),
                protect_first_n=3, protect_last_tokens=0,
                is_eligible=lambda r: True,
            )

        self.assertIsNotNone(result.summary_id)
        # Summary must be tagged as a doc chunk.
        s = eng._store.get_summary(result.summary_id)
        tags = json.loads(s.get("tags") or "[]")
        self.assertTrue(any(t.startswith("doc:") for t in tags), f"tags: {tags}")

        # Pin's start_hash must have advanced (or the pin is retired).
        remaining = [p for p in eng._store.list_pins() if p["id"] == pin_id]
        if remaining:
            self.assertNotEqual(remaining[0]["start_hash"], start_h,
                                "pin start_hash did not advance")
        eng.on_session_end("doc", [])

    def test_monolithic_doc_fully_folds_and_retires_pin(self) -> None:
        from plugins.autobio.compression import ModelInfo, compress_step
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()
        msgs = self._big_conversation(10, size=3000)
        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        start_h = message_hash(msgs[3])
        end_h = message_hash(msgs[7])
        pin_id = eng._store.add_pin(
            start_h, end_h, reason="static doc", kind="doc_monolithic",
            meta={"title": "ref"},
        )

        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("folded whole doc")):
            compress_step(
                msgs, eng._store, eng.cfg,
                ModelInfo(model="t"),
                protect_first_n=3, protect_last_tokens=0,
                is_eligible=lambda r: True,
            )

        remaining = [p for p in eng._store.list_pins() if p["id"] == pin_id]
        self.assertEqual(remaining, [], "monolithic pin should retire after full fold")
        eng.on_session_end("doc", [])

    def test_auto_detect_large_tool_result(self) -> None:
        """A tool result over threshold becomes a doc pin automatically."""
        eng = self._make_engine()
        eng.cfg["documents"]["auto_threshold_tokens"] = 500   # low for test
        eng.cfg["documents"]["auto_style"] = "chronological"

        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the log"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": json.dumps({"path": "/var/log/app.log"})}},
            ]},
            # 3000-char tool result → ~750 tokens → above threshold
            {"role": "tool", "tool_call_id": "t1", "name": "read_file",
             "content": "log line\n" * 400},
            {"role": "user", "content": "thanks"},
        ]
        eng._archive_live(msgs)

        docs = [p for p in eng._store.list_pins() if str(p.get("kind", "")).startswith("doc_")]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["kind"], "doc_chronological")
        meta = docs[0]["meta"]
        self.assertEqual(meta["source"], "auto")
        self.assertEqual(meta["tool_name"], "read_file")
        eng.on_session_end("doc", [])

    def test_auto_detect_respects_exclude_tools(self) -> None:
        eng = self._make_engine()
        eng.cfg["documents"]["auto_threshold_tokens"] = 100
        eng.cfg["documents"]["exclude_tools"] = ["memory_search"]

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "s1", "type": "function",
                 "function": {"name": "memory_search",
                              "arguments": json.dumps({"query": "x"})}},
            ]},
            {"role": "tool", "tool_call_id": "s1", "name": "memory_search",
             "content": "X" * 5000},
        ]
        eng._archive_live(msgs)

        docs = [p for p in eng._store.list_pins() if str(p.get("kind", "")).startswith("doc_")]
        self.assertEqual(len(docs), 0, "memory_search result should be excluded")
        eng.on_session_end("doc", [])

    def test_auto_detect_is_idempotent(self) -> None:
        """Re-archival of the same tool result does not create duplicate pins."""
        eng = self._make_engine()
        eng.cfg["documents"]["auto_threshold_tokens"] = 100

        msgs = [
            {"role": "tool", "tool_call_id": "t1", "name": "terminal",
             "content": "X" * 5000},
        ]
        eng._archive_live(msgs)
        eng._archive_live(msgs)
        eng._archive_live(msgs)

        docs = [p for p in eng._store.list_pins() if str(p.get("kind", "")).startswith("doc_")]
        self.assertEqual(len(docs), 1)
        eng.on_session_end("doc", [])

    def test_mark_and_unmark_document_via_cm(self) -> None:
        from plugins.autobio.cm import cm_respond
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()
        msgs = self._big_conversation(6, size=500)
        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        sh = message_hash(msgs[3])
        eh = message_hash(msgs[8])

        mark_reply = _fake_resp_with_ops(
            "OK, marking the log for you.",
            [f'MARK_DOCUMENT {sh} {eh} chronological "debug log"'],
        )
        with patch("agent.auxiliary_client.call_llm", return_value=mark_reply):
            reply = cm_respond("mark this log as a doc", eng, messages=msgs)
        self.assertIn("marked document pin", reply)

        docs = [p for p in eng._store.list_pins() if str(p.get("kind", "")).startswith("doc_")]
        self.assertEqual(len(docs), 1)
        pin_id = docs[0]["id"]

        unmark_reply = _fake_resp_with_ops(
            "Retiring the doc pin.",
            [f"UNMARK_DOCUMENT {pin_id}"],
        )
        with patch("agent.auxiliary_client.call_llm", return_value=unmark_reply):
            reply = cm_respond("unmark it", eng, messages=msgs)
        self.assertIn("unmarked document pin", reply)
        self.assertEqual(
            [p for p in eng._store.list_pins() if str(p.get("kind", "")).startswith("doc_")],
            [],
        )
        eng.on_session_end("doc", [])

    def test_boundary_snap_finds_paragraph(self) -> None:
        from plugins.autobio.compression import snap_to_natural_boundary

        text = "first paragraph body text here.\n\nsecond paragraph starts here."
        # Target somewhere inside the first paragraph; snap should land
        # after the blank-line break.
        snap = snap_to_natural_boundary(text, target_offset=20, window=50)
        self.assertGreater(snap, 20)
        # After the \n\n, text should begin with 'second'.
        self.assertTrue(text[snap:].lstrip().startswith("second"))


def _fake_resp_with_ops(reply_text: str, ops: List[str]) -> MagicMock:
    body = reply_text + "\n\nOPS:\n" + "\n".join(ops)
    return _resp(body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

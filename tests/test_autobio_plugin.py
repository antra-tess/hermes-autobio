"""End-to-end smoke tests for the autobio plugin.

Runs the engine + provider against a synthetic conversation large
enough to exercise: archival, tool-result pruning, L0→L1 compression,
L1→L2 merging, speculative cache hit, CM tool call, memory_search,
and /undo-equivalent rollback (hash-based cache invalidation).

These tests mock the auxiliary LLM so they run offline.
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

# Make sure the repo root is importable when pytest runs from tests/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _fake_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _build_conversation(n_turns: int, content_size: int = 1500) -> List[Dict[str, Any]]:
    """Synthetic alternating user/assistant messages."""
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User turn {i}: " + ("U" * content_size)})
        msgs.append({"role": "assistant", "content": f"Assistant {i}: " + ("A" * content_size)})
    return msgs


class AutobioE2E(unittest.TestCase):

    def setUp(self) -> None:
        # Pre-import so patch() can bind.
        import agent.auxiliary_client  # noqa: F401
        self.tmpdir = tempfile.mkdtemp(prefix="autobio_e2e_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_engine(self):
        from plugins.autobio.engine import AutobioContextEngine
        eng = AutobioContextEngine()
        eng.update_model(model="test-model", context_length=100_000)
        eng.on_session_start("e2e-session", hermes_home=self.tmpdir)
        return eng

    def test_compress_and_merge_full_flow(self) -> None:
        """3 distinct compressions auto-promote to an L2.

        Under archive-first semantics, summarize_chunk is idempotent by
        source_hash. To create 3 distinct L1s we must give each call a
        chunk with different content — same input twice is a cache hit.
        Speculation is disabled so the L1 count is deterministic.
        """
        eng = self._make_engine()
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 400
        eng.cfg["compression"]["merge_threshold"] = 3
        eng.cfg["compression"]["speculative"] = False
        # Rebuild speculative worker with the new config.
        if eng._speculative is not None:
            eng._speculative._enabled = False
        eng.threshold_tokens = 1_000

        summary_counter = [0]
        def fake_call(**kwargs):
            summary_counter[0] += 1
            return _fake_llm_response(f"Memory {summary_counter[0]}: things happened.")

        with patch("agent.auxiliary_client.call_llm", side_effect=fake_call):
            for i in range(3):
                # Distinctive content each iteration → distinct source_hash.
                msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
                for j in range(40):
                    msgs.append({"role": "user", "content": f"iter-{i}-u{j} " + ("U" * 3000)})
                    msgs.append({"role": "assistant", "content": f"iter-{i}-a{j} " + ("A" * 3000)})
                out = eng.compress(msgs)
                self.assertGreaterEqual(len(out), 1)

        l1_total = eng._store.list_summaries(1, include_merged=True)
        self.assertGreaterEqual(len(l1_total), 3, f"expected >= 3 L1s total, got {l1_total}")
        l2s = eng._store.list_summaries(2, include_merged=False)
        self.assertEqual(len(l2s), 1, f"expected 1 L2, got {l2s}")
        l1s_active = eng._store.list_summaries(1, include_merged=False)
        self.assertEqual(len(l1s_active), 0, "L1s should be merged")
        eng.on_session_end("e2e-session", [])

    def test_speculative_cache_hit(self) -> None:
        """Speculative worker result is reused by compress()."""
        eng = self._make_engine()
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 400
        eng.threshold_tokens = 1_000

        # Large enough to satisfy chunk + protect_last budget.
        msgs = _build_conversation(40, content_size=3000)
        llm_calls = []
        def fake_call(**kwargs):
            llm_calls.append(kwargs)
            return _fake_llm_response("Speculative memory written.")

        with patch("agent.auxiliary_client.call_llm", side_effect=fake_call):
            eng._speculative.schedule(msgs, protect_first_n=eng.protect_first_n)
            while eng._speculative._thread and eng._speculative._thread.is_alive():
                time.sleep(0.02)
            self.assertEqual(len(llm_calls), 1, "speculative worker did not call LLM")

            out = eng.compress(msgs)
            self.assertEqual(len(llm_calls), 1, "LLM called despite cache hit")

        self.assertLess(len(out), len(msgs))
        eng.on_session_end("e2e-session", [])

    def test_memory_search_and_grep(self) -> None:
        """Provider tools return results from the seeded archive."""
        from plugins.autobio.provider import AutobioMemoryProvider

        prov = AutobioMemoryProvider()
        prov.initialize("e2e-session", hermes_home=self.tmpdir)
        prov._store.insert_summary(
            level=1, source_hashes=["h1"], source_hash="h1",
            content="The auth_validator.check_token() returned 401 during tests.",
            tokens=80, compression_generations=1,
        )

        search_out = json.loads(prov.handle_tool_call("memory_search", {"query": "auth"}))
        self.assertGreaterEqual(len(search_out["results"]), 1)

        grep_out = json.loads(prov.handle_tool_call("memory_grep", {"pattern": r"check_token"}))
        self.assertGreaterEqual(len(grep_out["results"]), 1)

        read_out = json.loads(prov.handle_tool_call("memory_read", {"id": "L1-0001"}))
        self.assertIn("auth_validator", read_out["content"])

        prov.shutdown()

    def test_proactive_checkin_defer_and_inject(self) -> None:
        """Proactive CM respects DEFER and records INJECT as pending obs."""
        eng = self._make_engine()
        eng._store.kv_set("current_turn", 100)
        eng._store.kv_set("last_checkin_turn", 0)

        # DEFER path
        defer_resp = _fake_llm_response("DEFER 20 tool chain in progress")
        with patch("agent.auxiliary_client.call_llm", return_value=defer_resp):
            msg = eng.maybe_proactive_checkin(messages=[])
        self.assertIsNone(msg)
        self.assertEqual(eng._store.kv_get("deferred_until_turn"), 120)
        self.assertFalse(eng.checkin_due())

        # Advance past defer
        eng._store.kv_set("current_turn", 125)
        inject_resp = _fake_llm_response("INJECT\nI noticed we keep revisiting L1-0001 — want to pull it up?")
        with patch("agent.auxiliary_client.call_llm", return_value=inject_resp):
            msg = eng.maybe_proactive_checkin(messages=[])
        self.assertIsNotNone(msg)
        self.assertIn("L1-0001", msg["content"])
        # Counter advances
        self.assertEqual(eng._store.kv_get("last_checkin_turn"), 125)
        eng.on_session_end("e2e-session", [])

    def test_on_turn_end_triggers_speculation_and_checkin(self) -> None:
        """Per-turn hook schedules speculation and injects proactive CM.

        Uses a content-routing mock because speculation runs in a
        background thread and racing side_effects are flaky.
        """
        from plugins.autobio.cm import PROACTIVE_SYSTEM_PROMPT

        eng = self._make_engine()
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 400

        msgs = _build_conversation(40, content_size=3000)

        eng._store.kv_set("current_turn", 100)
        eng._store.kv_set("last_checkin_turn", 0)

        def routed_llm(**kwargs):
            # Route by system prompt: proactive CM vs. speculator.
            msgs_ = kwargs.get("messages") or []
            system_text = ""
            if msgs_ and msgs_[0].get("role") == "system":
                system_text = msgs_[0].get("content", "")
            if PROACTIVE_SYSTEM_PROMPT[:40] in system_text:
                return _fake_llm_response(
                    "INJECT\nNoticed the last few turns revisit auth — want L1-0001 surfaced?"
                )
            return _fake_llm_response("Speculative memory cached.")

        with patch("agent.auxiliary_client.call_llm", side_effect=routed_llm):
            eng.on_turn_end(msgs)
            while eng._speculative._thread and eng._speculative._thread.is_alive():
                time.sleep(0.02)

        # Proactive check-in was appended in-place.
        last = msgs[-1]
        self.assertEqual(last["role"], "assistant", f"last msg: {last}")
        self.assertIn("L1-0001", last["content"])

        # Under archive-first semantics, speculative summarization
        # writes directly to the summaries table, not a separate cache.
        l1s = eng._store.list_summaries(1, include_merged=False)
        self.assertGreaterEqual(len(l1s), 1, "speculative summary not written to archive")

        eng.on_session_end("e2e-session", msgs)

    def test_plugin_discovery(self) -> None:
        """Both halves are findable via the bundled loaders."""
        from plugins.context_engine import load_context_engine
        from plugins.memory import load_memory_provider

        eng = load_context_engine("autobio")
        self.assertIsNotNone(eng)
        self.assertEqual(eng.name, "autobio")

        prov = load_memory_provider("autobio")
        self.assertIsNotNone(prov)
        self.assertEqual(prov.name, "autobio")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

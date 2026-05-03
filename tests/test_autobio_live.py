"""Live API smoke test for the autobio plugin.

Exercises the paths that the mock suites cannot: real KV-preserving
summarization, real CM reasoning, real merge summaries. Uses the
Anthropic API via the auxiliary client.

**Not run by default.** Gated on ``AUTOBIO_LIVE_TEST=1`` in the
environment and requires ``ANTHROPIC_API_KEY`` set. The key must be
passed through the environment — never committed, never embedded in
this file.

To run:

    export ANTHROPIC_API_KEY=sk-ant-...          # fresh key from console
    export AUTOBIO_LIVE_TEST=1
    export AUTOBIO_LIVE_MODEL=claude-sonnet-4-5  # optional override
    python -m unittest tests.plugins.test_autobio_live -v

The test prints the actual LLM outputs so you can eyeball whether the
summaries and CM replies are sensible, not just structurally valid.
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_LIVE_ENABLED = os.getenv("AUTOBIO_LIVE_TEST") == "1"
_MODEL = os.getenv("AUTOBIO_LIVE_MODEL", "claude-sonnet-4-5")
_HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))


def _banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def _build_story(n_turns: int = 20) -> List[Dict[str, Any]]:
    """Synthetic conversation with recoverable details in it.

    The live summary should preserve the specific values (Redis,
    check_token, 401, /api/rate) so the CM recall test later can find
    them.
    """
    msgs: List[Dict[str, Any]] = [{
        "role": "system",
        "content": "You are a senior backend engineer helping the user refactor an auth system.",
    }]
    msgs.append({"role": "user", "content": "We need to pick a rate-limiting backend. Options on the table: Redis, in-memory, Postgres."})
    msgs.append({"role": "assistant", "content": "Redis makes the most sense — persistent, fast token-bucket support, survives restarts. Postgres would add write load."})
    msgs.append({"role": "user", "content": "Agreed, Redis. Now the auth bug: /api/rate returns 401 even for valid tokens."})
    msgs.append({"role": "assistant", "content": "Let me check auth_validator.check_token(). If the issuer claim isn't matching what our Redis cache stores, you'll get 401 on valid tokens."})
    msgs.append({"role": "user", "content": "Oh, iss is 'auth.prod.local' in Redis but the validator expects 'auth.prod'. Missing suffix."})
    msgs.append({"role": "assistant", "content": "That's your bug. Normalize the issuer at validation time — strip the suffix before comparing."})
    # Filler that will get compressed away but details should survive.
    # Each turn is ~2k chars of content so the total conversation clears
    # the (chunk + 2*chunk protect_last) budget the chunk selector needs.
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"Follow-up {i}: what about edge case E{i}? " + ("detail " * 300)})
        msgs.append({"role": "assistant", "content": f"Edge case E{i}: handle it by... " + ("reasoning " * 300)})
    msgs.append({"role": "user", "content": "Now, tell me again — what backend did we pick, and what was the 401 cause?"})
    return msgs


@unittest.skipUnless(_LIVE_ENABLED and _HAS_KEY,
                     "set AUTOBIO_LIVE_TEST=1 and ANTHROPIC_API_KEY to run")
class AutobioLive(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="autobio_live_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_engine(self):
        from plugins.autobio.engine import AutobioContextEngine
        eng = AutobioContextEngine()
        eng.update_model(
            model=_MODEL,
            context_length=200_000,
            provider="anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        eng.on_session_start("live", hermes_home=self.tmp)
        # Configure so compression fires on a medium-sized list.
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 500
        eng.cfg["compression"]["merge_threshold"] = 3
        # CM uses the same model for simplicity.
        eng.cfg["cm"]["model"] = _MODEL
        return eng

    # ----- Path 1: real KV-preserving L0 → L1 summarization ---------

    def test_01_real_summarization_preserves_facts(self) -> None:
        _banner("Path 1: real L0 → L1 summarization")
        eng = self._make_engine()
        msgs = _build_story(n_turns=15)

        out = eng.compress(msgs)
        l1s = eng._store.list_summaries(1, include_merged=False)
        self.assertEqual(len(l1s), 1)
        summary = l1s[0]["content"]
        print(f"\n--- L1-0001 summary ({l1s[0]['tokens']} tokens) ---")
        print(summary)
        print("------")

        # The summary should mention the key facts we seeded.
        s_lower = summary.lower()
        found_redis = "redis" in s_lower
        found_401 = "401" in summary or "unauthorized" in s_lower
        found_issuer = ("iss" in s_lower or "issuer" in s_lower or "suffix" in s_lower)
        print(f"facts preserved: redis={found_redis} 401={found_401} issuer={found_issuer}")
        # Don't hard-assert on every one — models vary. But at least
        # one of the core facts should survive.
        self.assertTrue(
            found_redis or found_401 or found_issuer,
            "summary lost all three seeded facts — KV-preserving prompt broken?",
        )

        # Message count dropped.
        self.assertLess(len(out), len(msgs))
        eng.on_session_end("live", out)

    # ----- Path 2: CM tool reasoning against real dashboard ---------

    def test_02_cm_answers_recall_question(self) -> None:
        _banner("Path 2: CM recall against real dashboard")
        eng = self._make_engine()
        # Seed a summary CM can find via SEARCH.
        eng._store.insert_summary(
            level=1, source_hashes=["h1"], source_hash="h1",
            content=(
                "We chose Redis for rate-limiting because it has persistence, "
                "fast token-bucket ops, and survives restarts. The 401 bug on "
                "/api/rate was caused by an issuer-suffix mismatch in "
                "auth_validator.check_token()."
            ),
            tokens=200, compression_generations=1,
        )

        reply = eng.handle_tool_call(
            "cm_manage_context",
            {"request": "What did we pick for rate limiting, and what caused the 401 bug?"},
            messages=[],
        )
        parsed = json.loads(reply)
        print("\n--- CM reply ---")
        print(parsed["reply"])
        print("------")
        text = parsed["reply"].lower()
        self.assertIn("redis", text, "CM failed to recall rate-limiting backend")
        self.assertTrue(
            "401" in parsed["reply"] or "issuer" in text or "suffix" in text,
            "CM failed to recall 401 cause",
        )
        eng.on_session_end("live", [])

    # ----- Path 3: real merge L1 → L2 -------------------------------

    def test_03_real_merge_summarizes_group(self) -> None:
        _banner("Path 3: real L1 → L2 merge")
        from plugins.autobio.compression import ModelInfo, merge_if_ready

        eng = self._make_engine()
        # Seed 3 distinct L1s so merge_threshold=3 triggers one L2.
        seed_contents = [
            "Turn 1-10: we designed the rate-limiting backend, picked Redis, drafted token-bucket config.",
            "Turn 11-20: we debugged the 401 bug on /api/rate — turned out to be issuer-suffix mismatch in auth_validator.check_token().",
            "Turn 21-30: we implemented the issuer-normalization fix and added regression tests in tests/auth/test_check_token.py.",
        ]
        for i, c in enumerate(seed_contents):
            eng._store.insert_summary(
                level=1, source_hashes=[f"h{i}"], source_hash=f"h{i}",
                content=c, tokens=80, compression_generations=1,
                first_turn=i * 10 + 1, last_turn=(i + 1) * 10,
            )

        mi = ModelInfo(
            model=_MODEL, provider="anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        created = merge_if_ready(eng._store, eng.cfg, mi)
        self.assertEqual(len(created), 1, f"expected one L2, got {created}")
        l2 = eng._store.get_summary(created[0])
        print(f"\n--- L2 {l2['id']} (gen {l2['compression_generations']}) ---")
        print(l2["content"])
        print("------")

        text = l2["content"].lower()
        # Merged summary should fold all three children.
        self.assertIn("redis", text)
        self.assertTrue("401" in l2["content"] or "issuer" in text)
        eng.on_session_end("live", [])

    # ----- Path 4: proactive INJECT with real CM --------------------

    def test_04_proactive_checkin_makes_sense(self) -> None:
        _banner("Path 4: real proactive CM check-in")
        eng = self._make_engine()
        # Make a dashboard with something CM might comment on.
        eng._store.insert_summary(
            level=1, source_hashes=["h1"], source_hash="h1",
            content="We pinned the rate-limiting debugging session (turns 40-55).",
            tokens=80, compression_generations=1,
        )
        eng._store.kv_set("current_turn", 100)
        eng._store.kv_set("last_checkin_turn", 0)

        msgs = _build_story(n_turns=5)
        msg = eng.maybe_proactive_checkin(msgs)
        print("\n--- CM check-in decision ---")
        if msg is None:
            print("DEFER")
            # A real model may reasonably defer — that's also a valid outcome.
        else:
            print(f"INJECT: {msg['content']}")
            self.assertEqual(msg["role"], "assistant")
            self.assertGreater(len(msg["content"]), 20)
        print("------")
        eng.on_session_end("live", msgs)

    # ----- Path 5: real doc_chronological fold ----------------------

    def test_05_doc_chronological_folds_top_with_as_of(self) -> None:
        """Verify with a real model that:
        - a chronological doc folds from the top under eligibility,
        - the summary is tagged doc:* and captures EARLY log content,
        - it does NOT leak LATE content (as-of invariant in practice),
        - the pin's start_hash advances after the fold.
        """
        _banner("Path 5: real doc_chronological fold + as-of preservation")
        from plugins.autobio.compression import ModelInfo, compress_step
        from plugins.autobio.ids import message_hash

        eng = self._make_engine()

        # Synthetic log spread across assistant messages. Each segment
        # carries a distinctive phrase so we can tell which slices
        # actually ended up in the summary. Each segment is padded to
        # ~1.4k tokens so chunk_tokens=4000 fits ~2-3 segments per fold.
        segments = [
            ("ALPHA",   "Deployment of service v2.3.1 begins at 12:00; 12 pods coming online."),
            ("BETA",    "First healthy request observed at 12:01:15; /api/health returns 200."),
            ("GAMMA",   "Redis connection pool exhausts; 503 errors spike on /api/rate."),
            ("DELTA",   "On-call engineer acknowledges alert; pod count scales from 12 to 20."),
            ("EPSILON", "503 rate stabilizes but p99 latency still elevated near 1.8 seconds."),
            ("ZETA",    "Root cause identified: pool max configured to 10 in config.yaml."),
            ("ETA",     "Hotfix v2.3.2 rolled out raising pool max to 50 across all pods."),
            ("THETA",   "All symptoms clear; incident marked resolved at 12:17, postmortem scheduled."),
        ]

        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are analyzing a production log."},
            {"role": "user", "content": "Reading the log now, in segments."},
        ]
        for tag, body in segments:
            msgs.append({
                "role": "assistant",
                "content": f"LOG {tag}: {body} " + ("detail " * 800),
            })
        msgs.append({"role": "user", "content": "Summarize what happened."})

        for i, m in enumerate(msgs):
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)
        first_log_idx = 2
        last_log_idx = 2 + len(segments) - 1
        start_h = message_hash(msgs[first_log_idx])
        end_h = message_hash(msgs[last_log_idx])
        pin_id = eng._store.add_pin(
            start_h, end_h, reason="prod log",
            kind="doc_chronological",
            meta={"title": "prod log", "original_tokens": 20_000},
        )

        mi = ModelInfo(
            model=_MODEL, provider="anthropic",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        result = compress_step(
            msgs, eng._store, eng.cfg, mi,
            protect_first_n=2, protect_last_tokens=0,
            is_eligible=lambda r: True,
        )
        self.assertIsNotNone(result.summary_id, "no summary produced")

        s = eng._store.get_summary(result.summary_id)
        print(f"\n--- doc L1 {result.summary_id} ({s['tokens']} tokens) ---")
        print(s["content"])
        print("------")

        tags = json.loads(s.get("tags") or "[]")
        print(f"tags: {tags}")
        self.assertTrue(
            any(t.startswith("doc:") for t in tags),
            f"summary not tagged as doc chunk; tags={tags}",
        )

        remaining = [p for p in eng._store.list_pins() if p["id"] == pin_id]
        if remaining:
            print(f"pin still active; start_hash was {start_h[:8]}, now {remaining[0]['start_hash'][:8]}")
            self.assertNotEqual(
                remaining[0]["start_hash"], start_h,
                "pin start_hash did not advance after fold",
            )
        else:
            print("pin retired — whole doc folded in one pass")

        s_lower = s["content"].lower()
        # Early markers (ALPHA/BETA content — always in the top chunk).
        early_hits = [
            x for x in ("v2.3.1", "deployment", "12 pods", "12:00", "12:01", "first")
            if x in s_lower
        ]
        # Late markers (ETA/THETA content — should never be in the top chunk).
        # Limited to phrases unique to later segments and immune to negation
        # ("not yet resolved" is a valid as-of statement, not a leak).
        late_hits = [
            x for x in ("v2.3.2", "12:17", "postmortem", "pool max to 50")
            if x in s_lower
        ]
        print(f"as-of: early_hits={early_hits}  late_hits={late_hits}")
        self.assertTrue(early_hits, "summary is missing all early-log markers")
        self.assertFalse(
            late_hits,
            f"future log content leaked into summary: {late_hits}",
        )

        eng.on_session_end("live", [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

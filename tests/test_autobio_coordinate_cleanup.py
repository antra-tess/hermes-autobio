"""Tests for the coordinate-cleanup refactor.

Exercises the four structural changes:

1. **Picker walks archive directly.** A mid-archive coverage gap must
   not produce a chunk that spans it.
2. **Recall pairs carry an explicit ``_autobio_recall`` flag.** The
   flag round-trips through render output and ``is_recall_message``
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

class RecallPairFlag(unittest.TestCase):
    """Both turns of a recall pair carry ``_autobio_recall = True`` and
    are recognized without the prefix fallback firing."""

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_rpf_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_recall_flag_survives_render_round_trip(self) -> None:
        """End-to-end: trigger a fold, get the rendered list, verify
        every recall-pair message has the flag, then re-feed the list
        through ``_reconcile_with_archive`` and verify the recall pair
        is skipped (no warning fired about prefix-only recognition).
        """
        import logging

        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.render import _RECALL_FLAG, is_recall_message

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("rpf", hermes_home=self.tmp)
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
                   return_value=_resp("memory body")):
            rendered = eng.compress(msgs)

        # Find recall-pair messages in the rendered output. Pairs come
        # in adjacent (header, body) couples — both should carry the
        # flag.
        recall_msgs = [m for m in rendered if is_recall_message(m)]
        self.assertGreater(
            len(recall_msgs), 0,
            "compress() should have produced at least one recall pair",
        )
        # Pairs come in 2s.
        self.assertEqual(
            len(recall_msgs) % 2, 0,
            f"recall messages should pair up; got odd count {len(recall_msgs)}",
        )
        for m in recall_msgs:
            self.assertIs(
                m.get(_RECALL_FLAG), True,
                f"recall message missing {_RECALL_FLAG} flag: {m}",
            )
        # Verify the alternation: even-indexed are user headers,
        # odd-indexed are assistant bodies.
        for i, m in enumerate(recall_msgs):
            expected = "user" if i % 2 == 0 else "assistant"
            self.assertEqual(
                m.get("role"), expected,
                f"recall pair role mismatch at idx {i}: expected "
                f"{expected}, got {m.get('role')}",
            )

        # Re-feed the rendered list. Reconcile should skip recall
        # messages without firing the prefix-fallback warning.
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
            if "prefix only" in r.getMessage()
        ]
        self.assertEqual(
            prefix_fallback, [],
            "prefix-fallback warning fired during round-trip "
            "(flag should have been the primary signal)",
        )

        eng.on_session_end("rpf", [])

    def test_prefix_fallback_fires_on_flagless_recall_header(self) -> None:
        """If a caller hand-builds a recall header without the flag
        (or the flag was stripped during a JSON round-trip), the
        prefix fallback should still recognize it AND log a warning
        so the gap is investigable.
        """
        import logging

        from plugins.autobio.render import is_recall_message

        flagless = {
            "role": "user",
            "content": "[CM] Recall memory L1-0001 (gen 1).",
        }

        autobio_logger = logging.getLogger("plugins.autobio.render")
        captured: List[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        cap = _Capture(level=logging.WARNING)
        autobio_logger.addHandler(cap)
        try:
            recognized = is_recall_message(flagless)
        finally:
            autobio_logger.removeHandler(cap)

        self.assertTrue(
            recognized,
            "prefix fallback should still recognize the recall header",
        )
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


# ---------------------------------------------------------------------------
# Tool-name enrichment (Read:ext suffix) + hierarchical weight lookup
# ---------------------------------------------------------------------------

class ToolNameEnrichment(unittest.TestCase):
    """Read-family tool results carry a ``:ext`` suffix on ``name`` so
    the picker's per-tool weight can distinguish documents from code.
    """

    def test_synthesize_tool_name_only_tags_read_family(self) -> None:
        from plugins.autobio.tool_naming import synthesize_tool_name

        # Read variants → tagged with extension.
        self.assertEqual(
            synthesize_tool_name("Read", {"file_path": "/x/y/foo.md"}),
            "Read:md",
        )
        self.assertEqual(
            synthesize_tool_name("read_file", {"path": "doc.rst"}),
            "read_file:rst",
        )
        # Read of an extensionless file → bare name (no false suffix).
        self.assertEqual(
            synthesize_tool_name("Read", {"file_path": "/etc/hosts"}),
            "Read",
        )
        self.assertEqual(
            synthesize_tool_name("Read", {}),
            "Read",
        )
        # Other tools pass through unchanged regardless of args.
        self.assertEqual(
            synthesize_tool_name("Bash", {"file_path": "/a.py"}),
            "Bash",
        )
        self.assertEqual(
            synthesize_tool_name("WebFetch", {"url": "x.com"}),
            "WebFetch",
        )

    def test_resolve_tool_weight_hierarchical(self) -> None:
        """Lookup tries exact match first, then strips the ``:ext`` and
        falls back to the bare tool, then to ``tool_default``."""
        from plugins.autobio.compression import _resolve_tool_weight

        cfg = {
            "tool_default": 0.5,
            "tool_by_name": {
                "Read": 0.1,
                "Read:md": 0.5,
                "Bash": 0.1,
            },
        }
        # Exact match wins.
        self.assertEqual(_resolve_tool_weight("Read:md", cfg), 0.5)
        # Suffix not in config → falls to bare prefix.
        self.assertEqual(_resolve_tool_weight("Read:py", cfg), 0.1)
        self.assertEqual(_resolve_tool_weight("Read:tex", cfg), 0.1)
        # Bare name → bare entry.
        self.assertEqual(_resolve_tool_weight("Read", cfg), 0.1)
        self.assertEqual(_resolve_tool_weight("Bash", cfg), 0.1)
        # Tool not mentioned → tool_default.
        self.assertEqual(_resolve_tool_weight("memory_search", cfg), 0.5)
        self.assertEqual(_resolve_tool_weight("anything:weird", cfg), 0.5)

    def test_archive_tags_read_with_extension(self) -> None:
        """End-to-end: archive an assistant+tool_result pair where the
        Read targets a markdown file, verify the stored row has
        ``name='Read:md'`` and that subsequent picker calls weight it
        at the document tier."""
        import tempfile, shutil
        from unittest.mock import patch
        from plugins.autobio.engine import AutobioContextEngine

        tmp = tempfile.mkdtemp(prefix="autobio_tne_")
        try:
            eng = AutobioContextEngine()
            eng.update_model(model="t", context_length=100_000)
            eng.on_session_start("tne", hermes_home=tmp)
            eng.cfg["compression"]["speculative"] = False
            if eng._speculative is not None:
                eng._speculative._enabled = False

            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": "read the spec"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {
                        "name": "Read",
                        "arguments": '{"file_path": "/repo/docs/spec.md"}',
                    }},
                ]},
                {"role": "tool", "tool_call_id": "c1", "name": "Read",
                 "content": "# Hierarchical Memory\n\nThis is the spec...\n"},
                {"role": "assistant", "content": "got it"},
                # And a code read for comparison.
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c2", "type": "function", "function": {
                        "name": "Read",
                        "arguments": '{"file_path": "/repo/src/main.py"}',
                    }},
                ]},
                {"role": "tool", "tool_call_id": "c2", "name": "Read",
                 "content": "import sys\nprint('hi')\n"},
            ]
            eng._archive_live(messages)

            import sqlite3
            conn = sqlite3.connect(f"{tmp}/memory/autobio/store.db")
            conn.row_factory = sqlite3.Row
            tool_rows = conn.execute(
                "SELECT tool_name, tool_call_id FROM messages "
                "WHERE session_id='tne' AND role='tool' "
                "ORDER BY position",
            ).fetchall()
            self.assertEqual(len(tool_rows), 2)
            names = {r["tool_call_id"]: r["tool_name"] for r in tool_rows}
            self.assertEqual(names["c1"], "Read:md")
            self.assertEqual(names["c2"], "Read:py")

            # Picker weight check: the Read:md row should weigh at 0.5,
            # Read:py at 0.1, under the production weights.
            from plugins.autobio.compression import _weighted_msg_tokens
            from plugins.autobio.config import load_autobio_config
            weights = load_autobio_config()["compression"]["weights"]
            doc_msg = {"role": "tool", "name": "Read:md", "content": "X" * 1000}
            code_msg = {"role": "tool", "name": "Read:py", "content": "X" * 1000}
            doc_w = _weighted_msg_tokens(doc_msg, weights)
            code_w = _weighted_msg_tokens(code_msg, weights)
            self.assertGreater(
                doc_w / max(1.0, code_w), 4.0,
                f"Read:md should weight at least 4x more than Read:py "
                f"(got doc={doc_w} code={code_w})",
            )

            eng.on_session_end("tne", [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ChunkAlignment(unittest.TestCase):
    """Picker must produce a usable chunk even when the first
    uncovered message is an oversized assistant + tool_calls."""

    def test_oversized_first_message_extends_forward(self) -> None:
        """A chunk that starts with a single assistant message whose
        tool_call arguments alone exceed ``chunk_token_target`` must
        not be discarded by alignment. The picker should extend the
        chunk forward through matching tool_results.

        Regression: the previous backward-align cut such chunks back
        to zero ('chunk ends with assistant+tool_calls' triggered the
        cut even when the message was the first and only one), so the
        picker returned no chunk and compression silently stopped.
        """
        import tempfile, shutil
        from plugins.autobio.compression import _pick_archive_chunk
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        tmp = tempfile.mkdtemp(prefix="autobio_align_")
        try:
            store = get_store(tmp, "align")

            # An assistant whose tool_call arguments alone are ~7k
            # tokens (above chunk_token_target=6000). Followed by a
            # matching tool result with normal content.
            big_args = '{"command": "' + "x" * 28_000 + '"}'
            asst = {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "c-big", "type": "function",
                    "function": {"name": "Bash", "arguments": big_args},
                }],
            }
            tool = {
                "role": "tool", "tool_call_id": "c-big", "name": "Bash",
                "content": "ok\n",
            }
            # Pad the tail with normal messages so the picker has
            # enough room to pick (else protect_last reservation
            # makes everything ineligible).
            extras = []
            for i in range(80):
                extras.append({"role": "user", "content": f"u{i} " + "y"*200})
                extras.append({"role": "assistant", "content": f"a{i} " + "z"*200})

            messages = [asst, tool] + extras
            for i, m in enumerate(messages):
                store.upsert_message(message_hash(m), m, turn=i, position=i)

            chunk_hashes, _ = _pick_archive_chunk(
                store,
                chunk_token_target=6000,
                # 0 reserve so we focus on the alignment bug, not on
                # tail-budget edge cases.
                protect_last_tokens=0,
                protect_first_n=0,
                weights_cfg={"default": 1.0, "tool_default": 1.0,
                             "tool_by_name": {"Bash": 1.0}},
            )
            self.assertGreater(
                len(chunk_hashes), 0,
                "picker should not return empty when chunk starts "
                "with an oversized assistant+tool_calls",
            )
            # The chunk must contain the matching tool message —
            # extending forward is the whole point.
            self.assertEqual(
                chunk_hashes[0], message_hash(asst),
                "chunk should start with the oversized assistant",
            )
            self.assertEqual(
                chunk_hashes[1], message_hash(tool),
                "chunk should include the matching tool result",
            )

            store.close()
            close_store(tmp, "align")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class AgentOriginRoleHonesty(unittest.TestCase):
    """Two hard rules govern role assignment in autobio prompts:

    1. **Bodies of agent-produced content** (memories, summaries —
       first-person recall the agent actually voiced) MUST be
       presented under ``role=assistant``. Wrapping them as user
       input would tell the model "the user said this happened"
       rather than "I recall this happening" — a meaningful KV-level
       distortion.

    2. **Headers / synthetic labels** (``MEMORY L1-...``, ``[autobio:
       memory ...]``, "[CM] Recall ...") are NOT produced by the
       LLM. They MUST go in ``role=user`` (CM's voice asking the
       agent to recall a memory) — putting them under assistant
       would falsely attribute system metadata to the agent.

    The result is a CM-asks / agent-recalls dialogue: each memory in
    a prompt is rendered as ``[user: header] [assistant: body]``.
    Anthropic dropped strict role-alternation, so consecutive
    same-role turns are fine — we don't need workarounds.
    """

    def test_render_recall_pair_role_split(self) -> None:
        """A recall pair has the header in user role (CM-voice) and
        the body in assistant role (the agent's first-person memory).
        """
        from plugins.autobio.render import _make_recall_pair
        pair = _make_recall_pair({
            "id": "L1-0001", "level": 1, "tokens": 800,
            "compression_generations": 1,
            "content": "I remember reading the spec carefully.",
        })
        self.assertEqual(len(pair), 2,
                         "recall pair should have exactly 2 turns")
        header, body = pair
        self.assertEqual(
            header.get("role"), "user",
            "recall header is CM-voice — must be user role",
        )
        self.assertIn("Recall memory L1-0001", header.get("content"))
        self.assertEqual(
            body.get("role"), "assistant",
            "recall body is agent-origin first-person — must be "
            "assistant role",
        )
        self.assertEqual(
            body.get("content"), "I remember reading the spec carefully.",
            "body content must match the L1's recall verbatim",
        )

    def test_legacy_compression_recall_pair_role_split(self) -> None:
        """The legacy ``compression._make_recall_pair`` (used by the
        list-based ``fold_chunk``) emits the same header/body shape
        as ``render._make_recall_pair``."""
        from plugins.autobio.compression import _make_recall_pair
        pair = _make_recall_pair({
            "id": "L1-0042",
            "compression_generations": 1,
            "content": "memory body content",
        })
        self.assertEqual(len(pair), 2)
        self.assertEqual(pair[0].get("role"), "user")
        self.assertEqual(pair[1].get("role"), "assistant")
        self.assertEqual(pair[1].get("content"), "memory body content")

    def test_summarizer_prompt_uses_dialogue_pattern(self) -> None:
        """Prior summaries shown to the summarizer must be split into
        a CM-voiced user header + assistant body pair, not a single
        wrapped 'MEMORY L1-X: <body>' message."""
        import tempfile, shutil
        from plugins.autobio.compression import _build_kv_preserving_prompt
        from plugins.autobio.store import close_store, get_store

        tmp = tempfile.mkdtemp(prefix="autobio_role_")
        try:
            store = get_store(tmp, "role")
            body = "I remember reading the spec."
            store.insert_summary(
                level=1, source_hashes=["h0"], source_hash="s0",
                content=body,
                tokens=100, compression_generations=1,
            )
            messages = [
                {"role": "user", "content": "go on"},
                {"role": "assistant", "content": "OK"},
            ]
            prompt = _build_kv_preserving_prompt(
                messages=messages, chunk_start=0, chunk_end=2,
                store=store, target_tokens=300,
            )

            # Find the header (user, says "Recall memory L1-...") and
            # immediately-following body (assistant, contains the recall).
            for i, m in enumerate(prompt):
                content = m.get("content") or ""
                if isinstance(content, str) and "Recall memory L1-" in content:
                    self.assertEqual(
                        m.get("role"), "user",
                        f"recall header must be user role: {m}",
                    )
                    # Next message should be assistant body.
                    self.assertLess(i + 1, len(prompt))
                    nxt = prompt[i + 1]
                    self.assertEqual(
                        nxt.get("role"), "assistant",
                        f"recall body must be assistant role: {nxt}",
                    )
                    self.assertEqual(nxt.get("content"), body)
                    break
            else:
                self.fail("no recall header found in summarizer prompt")
            store.close()
            close_store(tmp, "role")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_merge_prompt_uses_dialogue_pattern(self) -> None:
        """When merging L1s into L2, each child memory in the merge
        prompt is split into [user header + assistant body]."""
        from unittest.mock import MagicMock, patch
        import tempfile, shutil
        from plugins.autobio.compression import ModelInfo, merge_if_ready
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        tmp = tempfile.mkdtemp(prefix="autobio_merge_role_")
        try:
            store = get_store(tmp, "mr")
            for i in range(6):
                m = {"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"m{i}"}
                store.upsert_message(message_hash(m), m, turn=i, position=i)
            for k in range(3):
                hashes = [message_hash({"role": "user" if (2*k) % 2 == 0 else "assistant",
                                        "content": f"m{2*k}"}),
                          message_hash({"role": "user" if (2*k+1) % 2 == 0 else "assistant",
                                        "content": f"m{2*k+1}"})]
                sid = store.insert_summary(
                    level=1, source_hashes=hashes, source_hash=f"sh{k}",
                    content=f"first-person recall #{k}",
                    tokens=50, compression_generations=1,
                )
                store.mark_compressed(hashes, sid)

            captured_prompts = []
            def capture(**kw):
                captured_prompts.append(kw.get("messages") or [])
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "merged"
                return resp

            with patch("agent.auxiliary_client.call_llm", side_effect=capture):
                cfg = {"compression": {"merge_threshold": 3, "max_levels": 3,
                                       "summary_target_tokens": 100}}
                merge_if_ready(store, cfg, ModelInfo(model="t"))

            self.assertGreater(len(captured_prompts), 0)
            prompt = captured_prompts[0]

            # Every "Recall memory N/3 (L1-...)" header is user role,
            # and the very next message is the assistant body.
            found = 0
            for i, m in enumerate(prompt):
                c = m.get("content") or ""
                if isinstance(c, str) and "Recall memory" in c and "/" in c:
                    self.assertEqual(
                        m.get("role"), "user",
                        f"merge header must be user role: {m}",
                    )
                    self.assertLess(i + 1, len(prompt))
                    nxt = prompt[i + 1]
                    self.assertEqual(
                        nxt.get("role"), "assistant",
                        f"merge body must be assistant role: {nxt}",
                    )
                    self.assertIn("first-person recall", nxt.get("content"))
                    found += 1
            self.assertEqual(
                found, 3,
                f"expected 3 [user header / assistant body] pairs, got {found}",
            )

            store.close()
            close_store(tmp, "mr")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SummarizerPromptComposition(unittest.TestCase):
    """End-to-end composition of the summarizer prompt against a
    synthetic archive. Drives ``compress_step_archive`` with a mocked
    LLM and inspects the captured prompt structure to verify spec
    compliance:

    - Prior memories appear in archive-position order.
    - L2/L3 consolidations are EXPANDED to their constituent L1s in
      the prompt (the "fill lower orbitals first" / adaptive-
      resolution rule).
    - The ``[head — messages before the chunk]`` segment includes
      uncovered messages with position < the chunk's first position
      (protect_first_n head, pinned raw content, rollback debris).
    - The marker, chunk, and instruction follow in that order.
    - No content past the chunk leaks in (no tail_after_chunk).
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_compose_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_archive_with_covered_l1(
        self, store, n_covered_msgs: int, n_uncovered_msgs: int,
    ) -> "tuple[list[str], list[str]]":
        """Insert ``n_covered_msgs`` followed by ``n_uncovered_msgs``
        messages and mark the first batch as covered by a synthetic
        L1. Returns (covered_hashes, uncovered_hashes)."""
        from plugins.autobio.ids import message_hash
        covered_hashes: List[str] = []
        uncovered_hashes: List[str] = []
        pos = 0
        for i in range(n_covered_msgs):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"PASTBATCH_{i} " + ("x" * 200)}
            h = message_hash(m)
            store.upsert_message(h, m, turn=i, position=pos)
            covered_hashes.append(h)
            pos += 1
        l1_id = store.insert_summary(
            level=1, source_hashes=covered_hashes,
            source_hash="prior-batch-sh",
            content=f"I recall the first {n_covered_msgs} steps.",
            tokens=150, compression_generations=1,
        )
        store.mark_compressed(covered_hashes, l1_id)
        for i in range(n_uncovered_msgs):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role,
                 "content": f"uncovered-{i} " + ("y" * 300)}
            h = message_hash(m)
            store.upsert_message(h, m, turn=n_covered_msgs + i, position=pos)
            uncovered_hashes.append(h)
            pos += 1
        return covered_hashes, uncovered_hashes

    def test_head_includes_uncovered_messages_before_chunk(self) -> None:
        """A protect_first_n system message and any other uncovered
        position-before-the-chunk content must appear in head."""
        from plugins.autobio.compression import (
            ModelInfo, compress_step_archive,
        )
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "head_test")
        # System message at position 0 — uncovered, sits below all
        # subsequent compression. This is the typical
        # protect_first_n head.
        sys_msg = {"role": "user", "content": "INITIAL_USER_TURN: please help me"}
        sys_h = message_hash(sys_msg)
        store.upsert_message(sys_h, sys_msg, turn=0, position=0)
        # 30 messages of conversation, weighted enough that the
        # picker picks one chunk from positions 1..N (note the
        # picker has protect_first_n=1 by default).
        for i in range(1, 31):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role,
                 "content": f"conv-{i} " + ("z" * 400)}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        captured = []
        def capture(**kw):
            captured.append(kw.get("messages") or [])
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "I summarized the chunk."
            return r

        cfg = {
            "compression": {
                "chunk_tokens": 1_000,
                "summary_target_tokens": 200,
                "speculative": False,
                # Default weights — full weight for user/assistant.
            },
            "documents": {"eligibility": "size_pressure"},
        }
        with patch("agent.auxiliary_client.call_llm", side_effect=capture):
            compress_step_archive(
                store, cfg, ModelInfo(model="t"),
                protect_first_n=1, protect_last_tokens=500,
            )

        self.assertEqual(len(captured), 1, "expected one summarizer call")
        prompt = captured[0]
        # The system message body (INITIAL_USER_TURN: …) must appear
        # somewhere in the prompt — that's head.
        joined = "\n".join(
            (m.get("content") or "") if isinstance(m.get("content"), str) else ""
            for m in prompt
        )
        self.assertIn(
            "INITIAL_USER_TURN", joined,
            "head segment missing — first user message not in prompt",
        )
        store.close()
        close_store(self.tmp, "head_test")

    def test_head_empty_when_archive_is_clean(self) -> None:
        """If every message at position < chunk_first is covered by
        a summary, head is empty (no spurious entries)."""
        from plugins.autobio.compression import (
            ModelInfo, compress_step_archive,
        )
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "clean_head")
        self._build_archive_with_covered_l1(
            store, n_covered_msgs=10, n_uncovered_msgs=20,
        )

        captured = []
        def capture(**kw):
            captured.append(kw.get("messages") or [])
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "summary"
            return r

        cfg = {
            "compression": {
                "chunk_tokens": 1_000, "summary_target_tokens": 200,
                "speculative": False,
            },
            "documents": {"eligibility": "size_pressure"},
        }
        with patch("agent.auxiliary_client.call_llm", side_effect=capture):
            compress_step_archive(
                store, cfg, ModelInfo(model="t"),
                # protect_first_n=0 so the synthetic position-0
                # message is foldable along with the rest.
                protect_first_n=0, protect_last_tokens=500,
            )

        prompt = captured[0]
        # No "PASTBATCH_" raw content should appear in the prompt —
        # those messages are already represented by the L1 recall pair.
        for m in prompt:
            content = m.get("content") or ""
            if isinstance(content, str):
                self.assertNotIn(
                    "PASTBATCH_", content,
                    f"covered message leaked into prompt as raw: {m}",
                )
        store.close()
        close_store(self.tmp, "clean_head")

    def test_l2_consolidation_expanded_to_l1s_in_prompt(self) -> None:
        """When the archive has an L2 consolidating 6 L1s, the
        summarizer prompt must show all 6 L1s (lowest orbital first)
        rather than just the L2 marker. Spec: 'adaptive resolution',
        L1 = 'first-hand memory, high fidelity'."""
        from plugins.autobio.compression import (
            ModelInfo, _enumerate_l1s_for_prompt, compress_step_archive,
        )
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "l2_balance")

        # Build an archive: 18 messages, with positions 0..17.
        for i in range(18):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"m{i}-{'x'*30}"}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        # Manually create 3 L1s, each covering 6 messages, and merge
        # them into a single L2.
        l1_ids = []
        for k in range(3):
            hs = []
            for j in range(6):
                pos = k * 6 + j
                role = "user" if pos % 2 == 0 else "assistant"
                hs.append(message_hash({"role": role, "content": f"m{pos}-{'x'*30}"}))
            sid = store.insert_summary(
                level=1, source_hashes=hs, source_hash=f"sh-{k}",
                content=f"I remember chunk {k}: detail-{k}-alpha-beta",
                tokens=80, compression_generations=1,
            )
            store.mark_compressed(hs, sid)
            l1_ids.append(sid)

        l2_id = store.insert_summary(
            level=2, source_hashes=l1_ids, source_hash="l2-sh",
            content="Consolidated overview across 3 chunks.",
            tokens=120, compression_generations=2,
        )
        store.mark_summary_merged(l1_ids, l2_id)

        # Sanity: list_summaries(level=1, include_merged=False) is now
        # empty (all L1s merged into L2). list_summaries(level=2) has L2.
        self.assertEqual(
            len(store.list_summaries(1, include_merged=False)), 0,
        )
        self.assertEqual(
            len(store.list_summaries(2, include_merged=False)), 1,
        )

        # The prompt-enumeration helper must surface all 3 L1s
        # (despite them being merged) — that's the balance rule.
        l1s = _enumerate_l1s_for_prompt(store)
        self.assertEqual(
            len(l1s), 3,
            f"prompt should expand L2 to 3 L1s; got {[s['id'] for s in l1s]}",
        )

        # Drive a compress to see the actual prompt structure.
        # We need uncovered messages to form a new chunk; archive's
        # 18 messages are all covered. Add 30 more uncovered.
        for i in range(18, 48):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"newm{i}-{'y'*200}"}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        captured = []
        def capture(**kw):
            captured.append(kw.get("messages") or [])
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "new memory"
            return r

        cfg = {
            "compression": {
                "chunk_tokens": 1_000, "summary_target_tokens": 200,
                "speculative": False,
            },
            "documents": {"eligibility": "size_pressure"},
        }
        with patch("agent.auxiliary_client.call_llm", side_effect=capture):
            compress_step_archive(
                store, cfg, ModelInfo(model="t"),
                protect_first_n=0, protect_last_tokens=500,
            )

        self.assertEqual(len(captured), 1)
        prompt = captured[0]

        # Collect "Recall memory ..." headers — must be 3 L1s, NOT L2.
        recall_headers = [
            m for m in prompt
            if isinstance(m.get("content"), str)
            and "Recall memory" in m["content"]
        ]
        self.assertEqual(
            len(recall_headers), 3,
            f"expected 3 L1 recall headers, got {len(recall_headers)}: "
            f"{[m['content'] for m in recall_headers]}",
        )
        for h in recall_headers:
            self.assertIn("L1-", h["content"])
            self.assertNotIn(
                "L2-", h["content"],
                f"L2 leaked into prompt — should be expanded to L1s: {h}",
            )
        store.close()
        close_store(self.tmp, "l2_balance")

    def test_memories_ordered_by_position_in_prompt(self) -> None:
        """L1s appear in archive-position order, regardless of
        creation order or merge state."""
        from plugins.autobio.compression import _enumerate_l1s_for_prompt
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "order")
        # Insert messages at positions 0..29.
        for i in range(30):
            m = {"role": "user", "content": f"m{i}"}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        # Create 3 L1s in REVERSE order (cover late messages first).
        # List position order should still come out correct.
        l1c_hashes = [message_hash({"role": "user", "content": f"m{i}"})
                      for i in range(20, 30)]
        l1c = store.insert_summary(
            level=1, source_hashes=l1c_hashes, source_hash="shc",
            content="late",
            tokens=50, compression_generations=1,
        )
        store.mark_compressed(l1c_hashes, l1c)

        l1a_hashes = [message_hash({"role": "user", "content": f"m{i}"})
                      for i in range(0, 10)]
        l1a = store.insert_summary(
            level=1, source_hashes=l1a_hashes, source_hash="sha",
            content="early",
            tokens=50, compression_generations=1,
        )
        store.mark_compressed(l1a_hashes, l1a)

        l1b_hashes = [message_hash({"role": "user", "content": f"m{i}"})
                      for i in range(10, 20)]
        l1b = store.insert_summary(
            level=1, source_hashes=l1b_hashes, source_hash="shb",
            content="mid",
            tokens=50, compression_generations=1,
        )
        store.mark_compressed(l1b_hashes, l1b)

        # Enumerate — must come back ordered early, mid, late by
        # source position, NOT creation order (which was c, a, b).
        l1s = _enumerate_l1s_for_prompt(store)
        ids = [s["id"] for s in l1s]
        self.assertEqual(
            ids, [l1a, l1b, l1c],
            f"L1s should be in position order [early, mid, late], got {ids}",
        )
        store.close()
        close_store(self.tmp, "order")

    def test_no_tail_after_chunk_in_prompt(self) -> None:
        """Spec: nothing AFTER the chunk leaks into the summarizer
        prompt — that would be future information from the chunk's
        as-of perspective."""
        from plugins.autobio.compression import (
            ModelInfo, compress_step_archive,
        )
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "no_tail")
        # 60 messages, none covered. Chunk picker will fold a slice
        # from the start; the rest stays as live tail. Tail must NOT
        # appear in the prompt.
        # First half: foldable content with weighted tokens enough
        # to reach the chunk_target. Second half: clearly-marked
        # FUTURE_TAIL_* content with similar weight (so the tail
        # reservation is satisfied AND the picker has room to emit a
        # chunk from the front).
        for i in range(60):
            role = "user" if i % 2 == 0 else "assistant"
            content = (f"FUTURE_TAIL_{i} " + ("z" * 600) if i >= 30
                       else f"chunk-{i} " + ("z" * 600))
            m = {"role": role, "content": content}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        captured = []
        def capture(**kw):
            captured.append(kw.get("messages") or [])
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "summary"
            return r

        cfg = {
            "compression": {
                "chunk_tokens": 1_000, "summary_target_tokens": 200,
                "speculative": False,
            },
            "documents": {"eligibility": "size_pressure"},
        }
        with patch("agent.auxiliary_client.call_llm", side_effect=capture):
            compress_step_archive(
                store, cfg, ModelInfo(model="t"),
                protect_first_n=0, protect_last_tokens=2_000,
            )
        self.assertEqual(len(captured), 1)
        prompt = captured[0]

        # No "FUTURE_TAIL_*" content must appear anywhere in the
        # prompt — that's content past the chunk.
        for m in prompt:
            content = m.get("content") or ""
            if isinstance(content, str):
                self.assertNotIn(
                    "FUTURE_TAIL_", content,
                    f"post-chunk content leaked into prompt: {content[:80]!r}",
                )
        store.close()
        close_store(self.tmp, "no_tail")

    def test_full_prompt_structure(self) -> None:
        """End-to-end skeleton check: prompt is laid out as
        [memory recall pairs] [head] [marker] [chunk] [instruction],
        with no segments out of order.
        """
        from plugins.autobio.compression import (
            ModelInfo, compress_step_archive, _MARKER,
        )
        from plugins.autobio.ids import message_hash
        from plugins.autobio.store import close_store, get_store

        store = get_store(self.tmp, "skeleton")
        # System message at position 0 — head.
        sys_msg = {"role": "user", "content": "HEAD_SYSTEM_MARKER_TEXT"}
        store.upsert_message(message_hash(sys_msg), sys_msg, turn=0, position=0)

        # Synthetic L1 covering positions 1..10 — prior memory.
        l1_hashes = []
        for i in range(1, 11):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"old-{i}"}
            h = message_hash(m)
            store.upsert_message(h, m, turn=i, position=i)
            l1_hashes.append(h)
        l1 = store.insert_summary(
            level=1, source_hashes=l1_hashes, source_hash="prev",
            content="PRIOR_MEMORY_BODY content here",
            tokens=80, compression_generations=1,
        )
        store.mark_compressed(l1_hashes, l1)

        # New uncovered conversation — chunk territory.
        for i in range(11, 41):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role,
                 "content": f"CHUNK_BODY_{i} " + ("z" * 400)}
            store.upsert_message(message_hash(m), m, turn=i, position=i)

        captured = []
        def capture(**kw):
            captured.append(kw.get("messages") or [])
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "summary"
            return r

        cfg = {
            "compression": {
                "chunk_tokens": 1_000, "summary_target_tokens": 200,
                "speculative": False,
            },
            "documents": {"eligibility": "size_pressure"},
        }
        with patch("agent.auxiliary_client.call_llm", side_effect=capture):
            compress_step_archive(
                store, cfg, ModelInfo(model="t"),
                protect_first_n=1, protect_last_tokens=500,
            )

        prompt = captured[0]

        # Locate each landmark by content.
        idx_recall = next(
            (i for i, m in enumerate(prompt)
             if isinstance(m.get("content"), str)
             and "Recall memory L1-" in m["content"]),
            None,
        )
        idx_body = next(
            (i for i, m in enumerate(prompt)
             if isinstance(m.get("content"), str)
             and "PRIOR_MEMORY_BODY" in m["content"]),
            None,
        )
        idx_head = next(
            (i for i, m in enumerate(prompt)
             if isinstance(m.get("content"), str)
             and "HEAD_SYSTEM_MARKER_TEXT" in m["content"]),
            None,
        )
        idx_marker = next(
            (i for i, m in enumerate(prompt)
             if isinstance(m.get("content"), str)
             and m["content"] == _MARKER),
            None,
        )
        idx_chunk = next(
            (i for i, m in enumerate(prompt)
             if isinstance(m.get("content"), str)
             and "CHUNK_BODY_" in m["content"]),
            None,
        )
        idx_instr = next(
            (i for i, m in enumerate(reversed(prompt))
             if isinstance(m.get("content"), str)
             and "Write the memory" in m["content"]),
            None,
        )
        if idx_instr is not None:
            idx_instr = len(prompt) - 1 - idx_instr

        self.assertIsNotNone(idx_recall, "no recall header found")
        self.assertIsNotNone(idx_body, "prior memory body missing")
        self.assertIsNotNone(idx_head, "head missing")
        self.assertIsNotNone(idx_marker, "marker missing")
        self.assertIsNotNone(idx_chunk, "chunk content missing")
        self.assertIsNotNone(idx_instr, "instruction missing")

        # Skeleton order: recall_header < body < head < marker < chunk < instr
        self.assertLess(idx_recall, idx_body,
                        "recall header must precede its body")
        self.assertLess(idx_body, idx_head,
                        "memory body must precede head")
        self.assertLess(idx_head, idx_marker,
                        "head must precede marker")
        self.assertLess(idx_marker, idx_chunk,
                        "marker must precede chunk")
        self.assertLess(idx_chunk, idx_instr,
                        "chunk must precede instruction")

        store.close()
        close_store(self.tmp, "skeleton")


class MainLlmCallRecording(unittest.TestCase):
    """The autobio engine implements ``ContextEngine.on_llm_call`` to
    record the main agent's per-call request/response into the
    archive's ``llm_calls`` table. Without this, the viewer can only
    show plugin-internal LLM calls (summarizer/merge/CM); main-agent
    prompts — the actual memory-context rendering surface — would
    have nowhere to land for inspection.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_mlc_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_openai_shape_response(self) -> None:
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="haiku-4-5", context_length=200_000)
        eng.on_session_start("mlc1", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        api_kwargs = {
            "model": "anthropic/claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "[CM] Recall memory L1-0001 (gen 1)."},
                {"role": "assistant", "content": "I remember reading the spec."},
                {"role": "user", "content": "and now: please pick a number"},
            ],
            "max_tokens": 100,
        }
        # OpenAI-shape mock response.
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "I pick 7."
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.usage = MagicMock(
            prompt_tokens=2400, completion_tokens=8,
        )

        eng.on_llm_call(api_kwargs, response)

        calls = eng._store.list_llm_calls()
        main_calls = [c for c in calls if c["purpose"] == "main"]
        self.assertEqual(len(main_calls), 1)
        c = main_calls[0]
        self.assertEqual(c["model"], "anthropic/claude-haiku-4-5")
        self.assertEqual(c["response_text"], "I pick 7.")
        self.assertEqual(c["response_finish"], "stop")
        self.assertEqual(c["prompt_tokens"], 2400)
        self.assertEqual(c["completion_tokens"], 8)

        # The prompt is preserved verbatim.
        prompt = json.loads(c["prompt_json"])
        self.assertEqual(len(prompt), 3)
        self.assertIn("Recall memory L1-0001", prompt[0]["content"])

        eng.on_session_end("mlc1", [])

    def test_records_anthropic_native_shape(self) -> None:
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="claude", context_length=200_000)
        eng.on_session_start("mlc2", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        api_kwargs = {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
        }
        # Anthropic-native shape: content is a list of blocks, usage
        # uses input_tokens / output_tokens, finish via stop_reason.
        response = MagicMock(spec=["content", "stop_reason", "usage"])
        block = MagicMock(spec=["text"])
        block.text = "hello there"
        response.content = [block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(spec=["input_tokens", "output_tokens"])
        response.usage.input_tokens = 50
        response.usage.output_tokens = 3

        eng.on_llm_call(api_kwargs, response)

        calls = eng._store.list_llm_calls()
        main_calls = [c for c in calls if c["purpose"] == "main"]
        self.assertEqual(len(main_calls), 1)
        c = main_calls[0]
        self.assertEqual(c["response_text"], "hello there")
        self.assertEqual(c["response_finish"], "end_turn")
        self.assertEqual(c["prompt_tokens"], 50)
        self.assertEqual(c["completion_tokens"], 3)

        eng.on_session_end("mlc2", [])

    def test_records_failed_call_with_error(self) -> None:
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="any", context_length=200_000)
        eng.on_session_start("mlc3", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        api_kwargs = {
            "model": "x", "messages": [{"role": "user", "content": "go"}],
        }
        eng.on_llm_call(
            api_kwargs, response=None,
            error=RuntimeError("rate limited"),
        )

        calls = eng._store.list_llm_calls()
        main_calls = [c for c in calls if c["purpose"] == "main"]
        self.assertEqual(len(main_calls), 1)
        c = main_calls[0]
        self.assertIsNone(c["response_text"])
        self.assertIn("rate limited", c["error"])

        eng.on_session_end("mlc3", [])


class CmModelProviderResolution(unittest.TestCase):
    """The CM model name needs to match the resolved auxiliary
    provider. Bare ``claude-sonnet-4-6`` works on direct Anthropic
    but OpenRouter requires the ``anthropic/claude-sonnet-4-6``
    namespace. The autobio engine should normalize on the fly so the
    same default config works for both deployments.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_cm_provider_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_engine(self, *, base_url: str, provider: str):
        from plugins.autobio.engine import AutobioContextEngine
        eng = AutobioContextEngine()
        eng.update_model(
            model="anthropic/claude-haiku-4-5",
            context_length=200_000,
            base_url=base_url,
            api_key="test-key",
            provider=provider,
            api_mode="openai",
        )
        eng.cfg["compression"]["speculative"] = False
        eng.on_session_start("cm_prov", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False
        return eng

    def test_bare_claude_name_namespaced_for_openrouter(self) -> None:
        from plugins.autobio.cm import _resolve_cm_model_for_provider
        eng = self._make_engine(
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
        )
        try:
            cfg = {"model": "claude-sonnet-4-6"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "anthropic/claude-sonnet-4-6",
            )
            cfg = {"model": "claude-haiku-4-5"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "anthropic/claude-haiku-4-5",
            )
        finally:
            eng.on_session_end("cm_prov", [])

    def test_bare_claude_name_left_alone_for_anthropic_direct(self) -> None:
        from plugins.autobio.cm import _resolve_cm_model_for_provider
        eng = self._make_engine(
            base_url="https://api.anthropic.com",
            provider="anthropic",
        )
        try:
            cfg = {"model": "claude-sonnet-4-6"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "claude-sonnet-4-6",
            )
        finally:
            eng.on_session_end("cm_prov", [])

    def test_already_namespaced_passes_through(self) -> None:
        from plugins.autobio.cm import _resolve_cm_model_for_provider
        # Even on Anthropic-direct, an explicitly namespaced value
        # is honored verbatim (the user picked it on purpose).
        eng = self._make_engine(
            base_url="https://api.anthropic.com",
            provider="anthropic",
        )
        try:
            cfg = {"model": "anthropic/claude-sonnet-4-6"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "anthropic/claude-sonnet-4-6",
            )
            cfg = {"model": "openai/gpt-4o-mini"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "openai/gpt-4o-mini",
            )
        finally:
            eng.on_session_end("cm_prov", [])

    def test_non_claude_bare_name_passes_through(self) -> None:
        """The normalization is narrow — it only adjusts bare
        ``claude-*`` names. Other bare names are left alone (the
        adapter has no opinion on their correct namespacing)."""
        from plugins.autobio.cm import _resolve_cm_model_for_provider
        eng = self._make_engine(
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
        )
        try:
            cfg = {"model": "gpt-4o-mini"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "gpt-4o-mini",
            )
            cfg = {"model": "deepseek-chat"}
            self.assertEqual(
                _resolve_cm_model_for_provider(cfg, eng),
                "deepseek-chat",
            )
        finally:
            eng.on_session_end("cm_prov", [])

    def test_empty_or_none_passes_through(self) -> None:
        from plugins.autobio.cm import _resolve_cm_model_for_provider
        eng = self._make_engine(
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
        )
        try:
            self.assertIsNone(_resolve_cm_model_for_provider({}, eng))
            self.assertIsNone(
                _resolve_cm_model_for_provider({"model": None}, eng),
            )
            self.assertEqual(
                _resolve_cm_model_for_provider({"model": ""}, eng), "",
            )
        finally:
            eng.on_session_end("cm_prov", [])


class UncompressRefoldRoundTrip(unittest.TestCase):
    """``UNCOMPRESS <sid>`` and its inverse ``REFOLD <sid>`` toggle the
    summary's render state by writing to the ``unfolded_summaries``
    KV. Verified end-to-end by inspecting ``render_view`` output:
    folded → recall pair; unfolded → raw messages; refolded → recall
    pair again.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_unfold_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed_archive_with_one_l1(self, store) -> "tuple[str, list[str]]":
        """Insert 4 messages, fold them into a single L1. Returns
        ``(l1_id, source_hashes)``.
        """
        from plugins.autobio.ids import message_hash
        hashes = []
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"covered-msg-{i}"}
            h = message_hash(m)
            store.upsert_message(h, m, turn=i, position=i)
            hashes.append(h)
        l1_id = store.insert_summary(
            level=1, source_hashes=hashes, source_hash="sh-uf",
            content="I remember the four covered messages happened.",
            tokens=80, compression_generations=1,
        )
        store.mark_compressed(hashes, l1_id)
        return l1_id, hashes

    def test_uncompress_persists_unfold_state(self) -> None:
        from plugins.autobio.cm import _execute_ops
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.render import render_view, is_recall_message

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("uf1", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        l1_id, _ = self._seed_archive_with_one_l1(eng._store)

        # Folded baseline: render shows recall pair, no raw covered-msg-* content.
        view = render_view(eng._store, protect_first_n=0)
        self.assertEqual(
            sum(1 for m in view if is_recall_message(m)), 2,
            "expected one recall pair (header + body) for L1 in folded state",
        )
        joined = "\n".join(
            (m.get("content") or "") if isinstance(m.get("content"), str)
            else "" for m in view
        )
        self.assertNotIn("covered-msg-", joined)

        # UNCOMPRESS via the CM op handler.
        results = _execute_ops([f"UNCOMPRESS {l1_id}"], eng)
        self.assertEqual(len(results), 1)
        op_line, op_result = results[0]
        self.assertIn(f"uncompressed {l1_id}", op_result)

        # KV should now contain the L1.
        unfolded = eng._store.kv_get("unfolded_summaries", [])
        self.assertIn(l1_id, unfolded)

        # Subsequent render: NO recall pair for this L1; raw messages
        # come through instead.
        view2 = render_view(
            eng._store, protect_first_n=0,
            unfolded_summaries=set(unfolded),
        )
        recall_pairs_remaining = sum(
            1 for m in view2 if is_recall_message(m)
        )
        self.assertEqual(
            recall_pairs_remaining, 0,
            "L1's recall pair should have been replaced by raw messages",
        )
        joined2 = "\n".join(
            (m.get("content") or "") if isinstance(m.get("content"), str)
            else "" for m in view2
        )
        self.assertIn("covered-msg-0", joined2,
                      "raw source messages should now appear in the view")

        eng.on_session_end("uf1", [])

    def test_refold_reverts_to_recall_pair(self) -> None:
        from plugins.autobio.cm import _execute_ops
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.render import render_view, is_recall_message

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("uf2", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        l1_id, _ = self._seed_archive_with_one_l1(eng._store)

        # Unfold first.
        _execute_ops([f"UNCOMPRESS {l1_id}"], eng)
        self.assertIn(l1_id, eng._store.kv_get("unfolded_summaries", []))

        # REFOLD removes from KV.
        results = _execute_ops([f"REFOLD {l1_id}"], eng)
        self.assertEqual(len(results), 1)
        op_line, op_result = results[0]
        self.assertIn(f"refolded {l1_id}", op_result)
        self.assertNotIn(
            l1_id, eng._store.kv_get("unfolded_summaries", []),
        )

        # Render is back to recall-pair mode.
        view = render_view(
            eng._store, protect_first_n=0,
            unfolded_summaries=set(
                eng._store.kv_get("unfolded_summaries", []),
            ),
        )
        self.assertEqual(
            sum(1 for m in view if is_recall_message(m)), 2,
            "after REFOLD the L1 should render as a recall pair again",
        )
        eng.on_session_end("uf2", [])

    def test_uncompress_idempotent_refold_on_unfolded_no_op(self) -> None:
        from plugins.autobio.cm import _execute_ops
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("uf3", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        l1_id, _ = self._seed_archive_with_one_l1(eng._store)

        # UNCOMPRESS twice — KV doesn't gain a duplicate.
        _execute_ops([f"UNCOMPRESS {l1_id}"], eng)
        _execute_ops([f"UNCOMPRESS {l1_id}"], eng)
        unfolded = eng._store.kv_get("unfolded_summaries", [])
        self.assertEqual(unfolded.count(l1_id), 1)

        # REFOLD on a non-unfolded id reports no-op cleanly.
        results = _execute_ops(["REFOLD L1-9999"], eng)
        self.assertIn("not currently unfolded", results[0][1])

        eng.on_session_end("uf3", [])


class DashboardScope(unittest.TestCase):
    """The L1 row of CM's dashboard must include merged L1s so CM can
    UNCOMPRESS / REFOLD any L1 currently in the agent's prompt.
    Under the balance rule, render emits a recall pair for every L1
    in the archive — active OR merged into a parent.
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_dash_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_l1_dashboard_includes_merged(self) -> None:
        from plugins.autobio.cm import build_dashboard
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.ids import message_hash

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("dash", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        # Insert 12 messages, fold into 2 L1s of 6 each, then merge
        # into a single L2.
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"m{i}"}
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        l1_ids = []
        for k in range(2):
            hashes = []
            for j in range(6):
                pos = k * 6 + j
                role = "user" if pos % 2 == 0 else "assistant"
                hashes.append(message_hash({"role": role, "content": f"m{pos}"}))
            sid = eng._store.insert_summary(
                level=1, source_hashes=hashes, source_hash=f"sh-{k}",
                content=f"chunk-{k} recall body",
                tokens=80, compression_generations=1,
            )
            eng._store.mark_compressed(hashes, sid)
            l1_ids.append(sid)

        # Merge both L1s into an L2.
        l2_id = eng._store.insert_summary(
            level=2, source_hashes=l1_ids, source_hash="l2-sh",
            content="L2 consolidating chunks 0+1.",
            tokens=120, compression_generations=2,
        )
        eng._store.mark_summary_merged(l1_ids, l2_id)

        # Sanity: list_summaries(level=1, include_merged=False) is
        # empty (both L1s merged); the active layer count is 0 → 1
        # L2 active.
        self.assertEqual(
            eng._store.list_summaries(1, include_merged=False), [],
        )
        self.assertEqual(
            len(eng._store.list_summaries(2, include_merged=False)), 1,
        )

        # Dashboard must still surface the merged L1s in the l1 row.
        dashboard = build_dashboard(eng, messages=[])
        l1_layer = dashboard["layers"]["l1"]
        self.assertEqual(
            l1_layer["count"], 2,
            f"l1 dashboard count should include merged L1s; got "
            f"{l1_layer}",
        )
        recent_ids = [p["id"] for p in l1_layer["recent"]]
        for sid in l1_ids:
            self.assertIn(
                sid, recent_ids,
                f"merged L1 {sid} missing from dashboard previews",
            )

        # L2 stays active-only (the merged L1s above are L1, not L2).
        # If we ever merge L2→L3, only the active L2 should show in
        # l2.count — verify by reading the count.
        self.assertEqual(dashboard["layers"]["l2"]["count"], 1)
        eng.on_session_end("dash", [])


class L3MergeFormation(unittest.TestCase):
    """L2→L3 merge path is structurally identical to L1→L2 but takes
    a long conversation to exercise naturally (6 L2s = 36+ L1s with
    default merge_threshold=6). We use merge_threshold=2 to drive it
    in a small synthetic.

    Verifies:

    1. ``merge_if_ready`` runs L1→L2 then L2→L3 in the same call.
    2. The resulting L3 has source_hashes pointing at the L2s, which
       in turn have source_hashes pointing at the L1s, which point at
       message hashes — full chain back to source.
    3. compression_generations climbs (1 → 2 → 3) per level.
    4. Render still emits L1 recall pairs for the messages whose L1
       was rolled into an L2 then an L3 (balance rule: lower orbitals
       first, regardless of how high the merge tower climbs).
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_l3_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_l3_forms_from_4_l1s_with_threshold_2(self) -> None:
        from plugins.autobio.compression import merge_if_ready, ModelInfo
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.ids import message_hash

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("l3", hermes_home=self.tmp)
        if eng._speculative is not None:
            eng._speculative._enabled = False

        # Insert 16 messages and 4 L1s of 4 messages each, position-
        # contiguous. Default threshold=2 will: pair the L1s into 2
        # L2s, then pair those L2s into 1 L3 — all in one merge_if_ready
        # call (the function loops L1→L2 to exhaustion, then L2→L3).
        N_MESSAGES = 16
        L1_COUNT = 4
        MSGS_PER_L1 = N_MESSAGES // L1_COUNT

        for i in range(N_MESSAGES):
            role = "user" if i % 2 == 0 else "assistant"
            m = {"role": role, "content": f"m{i}"}
            eng._store.upsert_message(message_hash(m), m, turn=i, position=i)

        l1_ids = []
        for k in range(L1_COUNT):
            hashes = []
            for j in range(MSGS_PER_L1):
                pos = k * MSGS_PER_L1 + j
                role = "user" if pos % 2 == 0 else "assistant"
                hashes.append(message_hash({"role": role, "content": f"m{pos}"}))
            sid = eng._store.insert_summary(
                level=1, source_hashes=hashes, source_hash=f"l1-sh-{k}",
                content=f"L1 chunk {k} body",
                tokens=40, compression_generations=1,
            )
            eng._store.mark_compressed(hashes, sid)
            l1_ids.append(sid)

        eng.cfg["compression"]["merge_threshold"] = 2
        eng.cfg["compression"]["summary_target_tokens"] = 100
        model_info = ModelInfo(model="t")

        # Stub the auxiliary so merges return non-empty content.
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("merged content")):
            created = merge_if_ready(eng._store, eng.cfg, model_info)

        # Expected: 2 L2s + 1 L3 created.
        l2s = eng._store.list_summaries(level=2, include_merged=True)
        l3s = eng._store.list_summaries(level=3, include_merged=True)
        self.assertEqual(
            len(l2s), 2,
            f"expected 2 L2s after L1→L2; got {[s['id'] for s in l2s]}",
        )
        self.assertEqual(
            len(l3s), 1,
            f"expected 1 L3 after L2→L3; got {[s['id'] for s in l3s]}",
        )
        self.assertEqual(
            len(created), 3,
            f"merge_if_ready should report 3 new summaries; got {created}",
        )

        # Active layers: L1s all merged; L2s all merged; one active L3.
        self.assertEqual(
            eng._store.list_summaries(1, include_merged=False), [],
        )
        self.assertEqual(
            eng._store.list_summaries(2, include_merged=False), [],
        )
        active_l3 = eng._store.list_summaries(3, include_merged=False)
        self.assertEqual(len(active_l3), 1)

        l3 = l3s[0]
        # source_hashes on L3 → L2 ids (its direct children).
        l3_children = json.loads(l3["source_hashes"])
        self.assertEqual(
            sorted(l3_children),
            sorted(s["id"] for s in l2s),
            f"L3 children should be the two L2s; got {l3_children}",
        )

        # Walk one L2's source_hashes → should be L1 ids.
        l2_a = l2s[0]
        l2_a_children = json.loads(l2_a["source_hashes"])
        self.assertEqual(len(l2_a_children), 2)
        for child in l2_a_children:
            self.assertTrue(
                child.startswith("L1-"),
                f"L2 children should be L1 ids; got {child}",
            )

        # compression_generations: 1 (L1) → 2 (L2) → 3 (L3).
        self.assertEqual(int(l3["compression_generations"]), 3)
        self.assertEqual(int(l2_a["compression_generations"]), 2)

        # Balance rule: render still emits L1 recall pairs for the
        # messages whose L1 was rolled into an L2 then an L3 — render
        # walks compressed_into directly, so it always emits L1
        # fidelity regardless of how high the merge tower climbs.
        from plugins.autobio.render import render_view
        rendered = render_view(eng._store, protect_first_n=0)
        # Every L1 should produce a recall pair (header + body = 2 msgs).
        from plugins.autobio.render import is_recall_message
        recall_msgs = [m for m in rendered if is_recall_message(m)]
        self.assertEqual(
            len(recall_msgs), L1_COUNT * 2,
            f"balance rule: render should emit one L1 recall pair per "
            f"L1 ({L1_COUNT} pairs = {L1_COUNT * 2} msgs); got "
            f"{len(recall_msgs)} from rendered list of {len(rendered)}",
        )

        eng.on_session_end("l3", [])


class CrashResilience(unittest.TestCase):
    """Archive must remain consistent across an unclean shutdown.

    The risky window is between ``insert_summary`` (writes the L1 row)
    and the caller's subsequent ``mark_compressed`` (sets the source
    messages' ``compressed_into`` pointer). Both are autocommit, so a
    crash between them leaves an *orphan L1*: the summary exists, but
    no message is wired to it.

    Recovery hinges on ``find_summary_by_source_hash`` (called at the
    top of ``summarize_chunk``): on the next compress pass the orphan
    is rediscovered by its content-addressed source_hash and reused,
    after which the caller's ``mark_compressed`` finally runs.

    These tests verify:

    1. The crash window is observable (orphan exists, messages still
       NULL).
    2. The archive opens cleanly after the crash (no migration
       failure, no FTS5 corruption signal).
    3. The next compress() pass deduplicates onto the orphan instead
       of producing a second L1 with the same source content.
    4. Post-recovery, every ``messages.compressed_into`` points at an
       existing, non-merged summary (no dangling pointers).
    """

    def setUp(self) -> None:
        import agent.auxiliary_client  # noqa: F401
        self.tmp = tempfile.mkdtemp(prefix="autobio_crash_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_msgs(self) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
        for i in range(30):
            msgs.append({"role": "user",
                         "content": f"u{i} " + ("x" * 2500)})
            msgs.append({"role": "assistant",
                         "content": f"a{i} " + ("y" * 2500)})
        return msgs

    def test_orphan_summary_recovers_on_next_pass(self) -> None:
        """Crash between insert_summary and mark_compressed:
        next compress() pass deduplicates and completes the wiring.
        """
        from plugins.autobio.engine import AutobioContextEngine
        from plugins.autobio.store import Store

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        eng.on_session_start("crash1", hermes_home=self.tmp)
        eng.cfg["compression"]["chunk_tokens"] = 4_000
        eng.cfg["compression"]["summary_target_tokens"] = 300
        eng.cfg["compression"]["speculative"] = False
        if eng._speculative is not None:
            eng._speculative._enabled = False

        msgs = self._make_msgs()

        # Inject a crash: monkey-patch the store's mark_compressed to
        # raise on the first call. The summary will already have been
        # written by insert_summary at that point.
        store = eng._store
        original_mark = store.mark_compressed
        crash_state = {"raised": False}

        def crashing_mark(hashes, summary_id):
            if not crash_state["raised"]:
                crash_state["raised"] = True
                raise RuntimeError("simulated crash mid-mark_compressed")
            return original_mark(hashes, summary_id)

        store.mark_compressed = crashing_mark  # type: ignore[method-assign]

        # First compress: should write an orphan L1 and bubble the crash.
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("memory body")):
            with self.assertRaises(RuntimeError):
                eng.compress(msgs)

        # Invariant 1: orphan L1 exists in archive.
        l1s = store.list_summaries(level=1, include_merged=True)
        self.assertEqual(
            len(l1s), 1,
            f"expected exactly one orphan L1; got {len(l1s)}",
        )
        orphan_id = l1s[0]["id"]

        # Invariant 2: no message points at it yet — mark_compressed
        # raised before running.
        with store._lock:
            row = store._conn.execute(
                "SELECT COUNT(*) AS n FROM messages "
                "WHERE compressed_into = ? AND session_id = ?",
                (orphan_id, store.session_id),
            ).fetchone()
        self.assertEqual(
            row["n"], 0,
            "messages should not yet point at the orphan summary",
        )

        # Restore mark_compressed and run a second pass — this is the
        # "next compress() after restart" path.
        store.mark_compressed = original_mark  # type: ignore[method-assign]
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("memory body")):
            eng.compress(msgs)

        # Invariant 3: still exactly one L1 — find_summary_by_source_hash
        # deduplicated onto the orphan instead of writing a twin.
        l1s_after = store.list_summaries(level=1, include_merged=True)
        self.assertEqual(
            len(l1s_after), 1,
            f"recovery should reuse the orphan; got "
            f"{[s['id'] for s in l1s_after]}",
        )
        self.assertEqual(l1s_after[0]["id"], orphan_id)

        # Invariant 4: messages now point at the L1.
        with store._lock:
            row = store._conn.execute(
                "SELECT COUNT(*) AS n FROM messages "
                "WHERE compressed_into = ? AND session_id = ?",
                (orphan_id, store.session_id),
            ).fetchone()
        self.assertGreater(
            row["n"], 0,
            "second compress() should have wired messages to the L1",
        )

        eng.on_session_end("crash1", [])

    def test_archive_opens_cleanly_after_crash(self) -> None:
        """Simulate process exit during a half-finished compress: close
        the engine without graceful end, reopen via a fresh engine
        instance, run compress() again. Archive must come back online
        with an intact orphan L1 and pristine FTS5 indexes.
        """
        from plugins.autobio.engine import AutobioContextEngine

        # ---- session A: crash mid-compress -------------------------
        eng_a = AutobioContextEngine()
        eng_a.update_model(model="t", context_length=100_000)
        eng_a.on_session_start("crash2", hermes_home=self.tmp)
        eng_a.cfg["compression"]["chunk_tokens"] = 4_000
        eng_a.cfg["compression"]["summary_target_tokens"] = 300
        eng_a.cfg["compression"]["speculative"] = False
        if eng_a._speculative is not None:
            eng_a._speculative._enabled = False

        store_a = eng_a._store
        original_mark = store_a.mark_compressed
        crashed = {"yes": False}

        def crashing_mark(hashes, summary_id):
            if not crashed["yes"]:
                crashed["yes"] = True
                raise RuntimeError("simulated crash")
            return original_mark(hashes, summary_id)

        store_a.mark_compressed = crashing_mark  # type: ignore[method-assign]
        msgs = self._make_msgs()

        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("memory body")):
            with self.assertRaises(RuntimeError):
                eng_a.compress(msgs)

        # Drop the engine WITHOUT calling on_session_end. This
        # simulates a kill -9: the SQLite WAL still has uncommitted
        # writes for the in-flight transaction, but committed writes
        # (the orphan L1) are durable.
        del eng_a

        # Force a fresh Store handle by clearing the singleton cache.
        from plugins.autobio import store as store_mod
        store_mod._STORES.clear()

        # ---- session B: reopen, run compress() again ----------------
        eng_b = AutobioContextEngine()
        eng_b.update_model(model="t", context_length=100_000)
        eng_b.on_session_start("crash2", hermes_home=self.tmp)
        eng_b.cfg["compression"]["chunk_tokens"] = 4_000
        eng_b.cfg["compression"]["summary_target_tokens"] = 300
        eng_b.cfg["compression"]["speculative"] = False
        if eng_b._speculative is not None:
            eng_b._speculative._enabled = False

        store_b = eng_b._store

        # Archive came back online with the orphan present.
        l1s_b = store_b.list_summaries(level=1, include_merged=True)
        self.assertEqual(len(l1s_b), 1, "orphan L1 missing after reopen")

        # FTS5 query should not raise — sanity-check the archive isn't
        # corrupted in a way that breaks search.
        try:
            with store_b._lock:
                store_b._conn.execute(
                    "SELECT id FROM summaries_fts "
                    "WHERE summaries_fts MATCH 'memory' LIMIT 1"
                ).fetchall()
        except Exception as e:  # pragma: no cover — diagnostic
            self.fail(f"FTS5 query failed after reopen: {e}")

        # Re-run compress; recovery should wire up messages.
        with patch("agent.auxiliary_client.call_llm",
                   return_value=_resp("memory body")):
            eng_b.compress(msgs)

        l1s_final = store_b.list_summaries(level=1, include_merged=True)
        self.assertEqual(
            len(l1s_final), 1,
            f"reopen+recover produced duplicate L1s: "
            f"{[s['id'] for s in l1s_final]}",
        )

        # Invariant: no message points at a non-existent summary, and
        # no message points at a merged summary (would break unfold).
        with store_b._lock:
            dangling = store_b._conn.execute(
                "SELECT COUNT(*) AS n FROM messages m "
                "WHERE m.session_id = ? AND m.compressed_into IS NOT NULL "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM summaries s "
                "  WHERE s.id = m.compressed_into AND s.session_id = m.session_id"
                ")",
                (store_b.session_id,),
            ).fetchone()
        self.assertEqual(
            dangling["n"], 0,
            "messages.compressed_into points at a non-existent summary",
        )

        eng_b.on_session_end("crash2", [])


class PinOpHidden(unittest.TestCase):
    """PIN / UNPIN at the summary level are not exposed to CM in v1.

    Under the balance rule render emits every L1 (active or merged) as
    a recall pair regardless of ``summaries.pinned``, so summary-level
    pinning has no observable effect. We hide the ops to avoid the
    foot-gun where CM issues a "protection" the agent thinks is
    working.

    The op handler is kept (returns an explanatory error) so a CM that
    hallucinates the op from training data gets a clear signal rather
    than a silent success.
    """

    def test_pin_unpin_not_in_cm_system_prompt(self) -> None:
        from plugins.autobio.cm import CM_SYSTEM_PROMPT

        # The op names should not appear as instruction lines in the
        # available-operations list. Fail loud if they sneak back in.
        # We check for the canonical "  PIN " / "  UNPIN " indented
        # forms used in the op enumeration so we don't trip on the
        # word "pin" elsewhere (e.g. "MARK_DOCUMENT" doesn't contain
        # PIN as a token, but a future doc string might mention pins).
        self.assertNotIn("\n  PIN ", CM_SYSTEM_PROMPT)
        self.assertNotIn("\n  UNPIN ", CM_SYSTEM_PROMPT)

    def test_pin_op_returns_explanatory_error(self) -> None:
        from plugins.autobio.cm import _execute_ops
        from plugins.autobio.engine import AutobioContextEngine

        eng = AutobioContextEngine()
        eng.update_model(model="t", context_length=100_000)
        tmp = tempfile.mkdtemp(prefix="autobio_pinhide_")
        try:
            eng.on_session_start("pinhide", hermes_home=tmp)
            results = _execute_ops(["PIN L1-0001"], eng)
            self.assertEqual(len(results), 1)
            # Result should explain the op is unavailable, not silently
            # succeed.
            _, msg = results[0]
            self.assertIn("not available", msg.lower())
            results = _execute_ops(["UNPIN L1-0001"], eng)
            _, msg = results[0]
            self.assertIn("not available", msg.lower())
            eng.on_session_end("pinhide", [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

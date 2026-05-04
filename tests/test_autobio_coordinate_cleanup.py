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

    def test_render_marker_is_user_role_header(self) -> None:
        """The live-view marker is a header-only artifact (the body
        lives in the archive). Headers are CM-voice, so user role.
        """
        from plugins.autobio.render import _make_marker
        marker = _make_marker({
            "id": "L1-0001", "level": 1, "tokens": 800,
            "compression_generations": 1, "content": "x",
        })
        self.assertEqual(
            marker.get("role"), "user",
            "live-view marker is a header from CM, not agent recall",
        )

    def test_legacy_summary_marker_is_user_role_header(self) -> None:
        from plugins.autobio.compression import _make_summary_marker
        marker = _make_summary_marker("L1-0042", "memory body", level=1)
        self.assertEqual(marker.get("role"), "user")

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


if __name__ == "__main__":
    unittest.main()

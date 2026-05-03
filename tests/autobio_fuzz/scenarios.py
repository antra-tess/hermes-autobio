"""Curated hazard scenarios.

Each function returns a list of ``Turn`` steps targeting a specific
structural boundary. When the harness discovers a bug anywhere, add a
reduced repro here so regressions get caught.

Naming: ``scenario_<what_hazard>``. Each docstring states the invariant
being protected.
"""

from __future__ import annotations

from typing import List

from .steps import (
    AddAssistant, AddToolCall, AddUser,
    MarkPin, Rollback, Step, TriggerCompress, Turn, UnmarkPin,
)


def scenario_pin_straddles_tool_pair() -> List[Step]:
    """Pin includes an assistant+tool_result pair. After fold (of
    anything else), the pair must remain paired in the live view."""
    turns: List[Turn] = [
        Turn([AddUser("setup")]),
        Turn([AddAssistant("ok")]),
        Turn([AddToolCall(
            tool_name="terminal",
            arguments={"command": "ls"},
            result="file1\nfile2\n",
            call_id="c-straddle",
        )]),
        Turn([AddAssistant("saw files")]),
        # Pin across the assistant+tool pair.
        Turn([MarkPin(start_index=2, end_index=4, kind="pin",
                      reason="straddle test")]),
    ]
    # Grow the tail so compression is motivated.
    for i in range(30):
        turns.append(Turn([AddUser(f"ramble {i} " + "x" * 600)]))
        turns.append(Turn([AddAssistant(f"response {i} " + "y" * 600)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_rollback_into_speculated_chunk() -> List[Step]:
    """Speculative worker writes an L1, then /undo reaches into the
    chunk. A fresh L1 must be produced on the next compress; both
    summaries should survive in the archive."""
    turns: List[Turn] = [Turn([AddUser("kickoff")])]
    for i in range(40):
        turns.append(Turn([AddUser(f"early-{i} " + "a" * 500)]))
        turns.append(Turn([AddAssistant(f"early-reply-{i} " + "b" * 500)]))
    # Force a compress to land an L1.
    turns.append(Turn([TriggerCompress()]))
    # Now reach back into the early turns and rewrite one.
    turns.append(Turn([Rollback(n=70)]))
    turns.append(Turn([AddUser("rewritten-early " + "z" * 500)]))
    for i in range(30):
        turns.append(Turn([AddAssistant(f"post-rollback-{i} " + "c" * 500)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_compress_mid_tool_loop() -> List[Step]:
    """Compression fires while a tool_call is open (assistant emitted
    the call but the tool result hasn't landed yet). The returned
    live list must not strand the open call across a user turn."""
    turns: List[Turn] = [Turn([AddUser("start")])]
    for i in range(35):
        turns.append(Turn([AddUser(f"u-{i} " + "p" * 500)]))
        turns.append(Turn([AddAssistant(f"a-{i} " + "q" * 500)]))
    # Emit an unpaired assistant tool_call (generator trick: add an
    # AddToolCall with an empty result string, then the harness turns
    # it into just the opening call). The cleanest way is to add a
    # manual Turn with AddAssistant plus a fake tool-call structure,
    # but our ops don't support that directly — so we use Trigger
    # before the result lands:
    turns.append(Turn([
        AddToolCall(
            tool_name="terminal",
            arguments={"command": "sleep 1"},
            result="",
            call_id="mid-loop",
        ),
        TriggerCompress(),
    ]))
    return turns


def scenario_overlapping_pins() -> List[Step]:
    """Two pins with overlapping ranges. The chunk picker must not
    loop or skip both; the final tail must stay valid."""
    turns: List[Turn] = [Turn([AddUser("setup")])]
    for i in range(20):
        turns.append(Turn([AddAssistant(f"think {i} " + "m" * 500)]))
    turns.append(Turn([MarkPin(start_index=3, end_index=9, kind="pin")]))
    turns.append(Turn([MarkPin(start_index=7, end_index=14, kind="pin")]))
    for i in range(20):
        turns.append(Turn([AddUser(f"more {i} " + "n" * 500)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_pin_at_head() -> List[Step]:
    """Pin starts at protect_first_n boundary. Chunk picker must not
    try to fold inside the protected head region."""
    turns: List[Turn] = [
        Turn([AddUser("head-1")]),
        Turn([AddAssistant("head-2")]),
        Turn([AddUser("head-3")]),
        Turn([MarkPin(start_index=1, end_index=3, kind="pin",
                      reason="head pin")]),
    ]
    for i in range(40):
        turns.append(Turn([AddUser(f"body-{i} " + "x" * 500)]))
        turns.append(Turn([AddAssistant(f"body-a-{i} " + "y" * 500)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_doc_top_chunk_off_by_one() -> List[Step]:
    """A doc whose first chunk is exactly chunk_tokens minus a
    handful. Tests boundary snapping when the picker has no slack."""
    turns: List[Turn] = [Turn([AddUser("read the log")])]
    # Single tool result, sized to ~3 chunks worth. Larger docs aren't
    # useful here and cause runaway retries when the summarizer has
    # reasoning-model quirks.
    turns.append(Turn([AddToolCall(
        tool_name="read_file",
        arguments={"path": "/var/log/edge.log"},
        result="\n".join(f"line-{i} " + "p" * 12 for i in range(800)),
        call_id="doc-1",
    )]))
    # Grow the tail a bit so doc eligibility fires under pressure.
    for i in range(30):
        turns.append(Turn([AddUser(f"follow {i} " + "s" * 500)]))
        turns.append(Turn([AddAssistant(f"resp {i} " + "t" * 500)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_rollback_then_readvance() -> List[Step]:
    """Rollback, re-add equivalent content, verify hashes line up and
    the archive does not double-count."""
    turns: List[Turn] = [Turn([AddUser("start")])]
    for i in range(30):
        turns.append(Turn([AddUser(f"m-{i} " + "q" * 400)]))
        turns.append(Turn([AddAssistant(f"r-{i} " + "w" * 400)]))
    turns.append(Turn([Rollback(n=4)]))
    # Re-add with slightly different content so new hashes exist.
    for i in range(4):
        turns.append(Turn([AddUser(f"re-{i} " + "z" * 400)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_unpin_of_retired_doc() -> List[Step]:
    """Doc auto-retires after full fold, then we issue UnmarkPin for
    its id. Harness must tolerate the stale reference cleanly."""
    turns: List[Turn] = [Turn([AddUser("read")])]
    turns.append(Turn([AddToolCall(
        tool_name="read_file",
        arguments={"path": "/tmp/big.log"},
        result="\n".join(f"entry-{i}" for i in range(500)),
        call_id="doc-u",
    )]))
    for i in range(25):
        turns.append(Turn([AddUser(f"chat {i} " + "u" * 600)]))
        turns.append(Turn([AddAssistant(f"reply {i} " + "v" * 600)]))
    # Force multiple compresses so the whole doc folds away.
    for _ in range(3):
        turns.append(Turn([TriggerCompress()]))
    # After retire, try to unpin the original range (start/end_index
    # no longer present in live list — harness resolve_unmark must
    # skip with a warning, not crash).
    turns.append(Turn([UnmarkPin(start_index=1, end_index=1,
                                 kind="doc_chronological")]))
    return turns


def scenario_two_docs_back_to_back() -> List[Step]:
    """Two auto-detected docs adjacent in the list. Chunk picker
    must navigate the boundary between them correctly."""
    turns: List[Turn] = [Turn([AddUser("read two logs")])]
    for j in range(2):
        turns.append(Turn([AddToolCall(
            tool_name="read_file",
            arguments={"path": f"/tmp/log_{j}.log"},
            result="\n".join(f"log-{j}-line-{i}" for i in range(400)),
            call_id=f"doc-{j}",
        )]))
    for i in range(40):
        turns.append(Turn([AddUser(f"between {i} " + "a" * 500)]))
        turns.append(Turn([AddAssistant(f"reply {i} " + "b" * 500)]))
    turns.append(Turn([TriggerCompress()]))
    return turns


def scenario_tool_result_exactly_at_threshold() -> List[Step]:
    """Tool result sized at ~auto_threshold_tokens ±epsilon. Verify
    the off-by-one around the detect boundary doesn't flap."""
    turns: List[Turn] = [Turn([AddUser("probe threshold")])]
    # ~50k tokens default → ~200_000 chars; we go slightly above/below.
    turns.append(Turn([AddToolCall(
        tool_name="read_file",
        arguments={"path": "/tmp/just_under.txt"},
        result="x" * (50_000 * 4 - 200),
        call_id="thr-1",
    )]))
    turns.append(Turn([AddToolCall(
        tool_name="read_file",
        arguments={"path": "/tmp/just_over.txt"},
        result="y" * (50_000 * 4 + 200),
        call_id="thr-2",
    )]))
    for i in range(20):
        turns.append(Turn([AddUser(f"q {i}"), AddAssistant(f"a {i}")]))
    turns.append(Turn([TriggerCompress()]))
    return turns


ALL_SCENARIOS = [
    ("pin_straddles_tool_pair",          scenario_pin_straddles_tool_pair),
    ("rollback_into_speculated_chunk",   scenario_rollback_into_speculated_chunk),
    ("compress_mid_tool_loop",           scenario_compress_mid_tool_loop),
    ("overlapping_pins",                 scenario_overlapping_pins),
    ("pin_at_head",                      scenario_pin_at_head),
    ("doc_top_chunk_off_by_one",         scenario_doc_top_chunk_off_by_one),
    ("rollback_then_readvance",          scenario_rollback_then_readvance),
    ("unpin_of_retired_doc",             scenario_unpin_of_retired_doc),
    ("two_docs_back_to_back",            scenario_two_docs_back_to_back),
    ("tool_result_exactly_at_threshold", scenario_tool_result_exactly_at_threshold),
]

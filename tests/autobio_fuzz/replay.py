"""Test-side adapter: load a Claude Code transcript and turn it into
``Step`` sequences the fuzz harness can execute.

The actual transcript parsing lives in ``plugins.autobio.ingest`` —
this module is just the bridge between the canonical
``IngestedMessage`` stream and the harness's ``Turn``/``AddX`` step
types.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from plugins.autobio.ingest import ClaudeCodeAdapter, IngestedMessage, TranscriptAdapter

from .steps import (
    AddAssistant, AddToolCall, AddUser, Step, Turn,
)

logger = logging.getLogger(__name__)


def load_claude_jsonl(
    path: str | Path,
    *,
    max_records: Optional[int] = None,
    include_system_reminders: bool = True,
) -> List[Step]:
    """Convenience wrapper: build a ClaudeCodeAdapter and convert."""
    adapter = ClaudeCodeAdapter(
        path=Path(path),
        max_records=max_records,
        include_system_reminders=include_system_reminders,
    )
    return adapter_to_steps(adapter)


def adapter_to_steps(adapter: TranscriptAdapter) -> List[Step]:
    """Convert any adapter's stream into harness Step turns.

    Messages with the same ``turn`` number are grouped into a single
    ``Turn`` so the harness fires ``on_turn_end`` at the same cadence
    the source instance saw. This matters for speculative timing and
    proactive CM scheduling.
    """
    by_turn: dict[int, List[IngestedMessage]] = {}
    for msg in adapter.iter_messages():
        by_turn.setdefault(msg.turn, []).append(msg)

    turns: List[Turn] = []
    for turn_num in sorted(by_turn):
        substeps: List = []
        msgs = by_turn[turn_num]
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m.role == "assistant" and m.tool_calls:
                # An assistant record may carry N parallel tool_calls.
                # The harness models tool calls one at a time
                # (AddToolCall), so we emit one per call and pair each
                # with its matching tool result by tool_call_id.
                import json
                for tc in m.tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tu_id = tc.get("id")
                    fn = tc.get("function") or {}
                    tool_name = fn.get("name") or "unknown"
                    arg_str = fn.get("arguments") or "{}"
                    try:
                        args = (
                            json.loads(arg_str)
                            if isinstance(arg_str, str) else arg_str
                        )
                    except Exception:
                        args = {}
                    # Find the matching tool result (search the whole
                    # turn — results may follow in any order).
                    result_idx: Optional[int] = None
                    for j in range(i + 1, len(msgs)):
                        if (msgs[j].role == "tool"
                                and msgs[j].tool_call_id == tu_id):
                            result_idx = j
                            break
                    if result_idx is not None:
                        result_msg = msgs[result_idx]
                        result_text = (
                            result_msg.content
                            if isinstance(result_msg.content, str)
                            else str(result_msg.content)
                        )
                        substeps.append(AddToolCall(
                            tool_name=tool_name,
                            arguments=args if isinstance(args, dict) else {},
                            result=result_text,
                            call_id=tu_id or f"unknown_{turn_num}_{i}",
                        ))
                        msgs[result_idx] = IngestedMessage(
                            role="_consumed", content="", turn=turn_num,
                        )
                    else:
                        logger.debug(
                            "adapter_to_steps: assistant tool_call %s on "
                            "turn %d has no matching result — dropped",
                            tu_id, turn_num,
                        )
            elif m.role == "assistant":
                content = m.content if isinstance(m.content, str) else str(m.content)
                if content.strip():
                    substeps.append(AddAssistant(content))
            elif m.role == "user":
                content = m.content if isinstance(m.content, str) else str(m.content)
                if content.strip():
                    substeps.append(AddUser(content))
            elif m.role == "tool":
                # Standalone tool result (no preceding assistant
                # tool_call in this turn). Skip — adding a tool message
                # without a matching call would break the live list.
                pass
            # role == "_consumed" is the marker we set above; skip
            i += 1
        if substeps:
            turns.append(Turn(substeps))
    return turns

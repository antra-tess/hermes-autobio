"""Claude Code session JSONL adapter.

A Claude Code session is one JSONL file at
``~/.claude/projects/<project-slug>/<session-uuid>.jsonl``. Each line is
a record with ``type`` ∈ {user, assistant, system, attachment} and a
``message.content`` that is either a string or a list of typed content
blocks (text / tool_use / tool_result / thinking / image).

Adapter responsibilities:

- Pair ``tool_use`` blocks with their later ``tool_result`` blocks so
  the engine sees a well-formed assistant-with-tool_calls + tool
  result pair, not split across turns.
- Carry a ``turn`` counter that increments on each user-initiated turn
  (so derived turn ranges in the archive line up with how a human
  would describe "what happened around turn 50").
- Preserve ``thinking`` and ``redacted_thinking`` content blocks on
  assistant messages by attaching them to ``IngestedMessage.reasoning_details``.
  Hermes' anthropic_adapter reads ``reasoning_details`` back when the
  message is sent to Claude again, so signatures stay valid and the
  KV state of the replayed conversation matches the original.
- Preserve ``image`` content blocks as structured content. If a user
  or assistant message contains image attachments, the resulting
  ``IngestedMessage.content`` is a list of blocks (mixing text and
  image dicts), not a flattened string. The autobio archive stores
  list content as JSON, so structure round-trips.

Skipped:

- ``thinking`` text is preserved (see above) but not surfaced as a
  separate text content block — the actual conversation content is
  the assistant's narration / tool calls, not its private monologue.
- ``type: "system"`` records (framework setup, not conversational).
- ``type: "attachment"`` records — these are NOT user-uploaded files.
  They're framework metadata: deferred-tool deltas, MCP server load
  events, hook outputs, etc. User-uploaded files appear as
  ``image`` (or future file-type) content blocks INSIDE user records,
  which we DO preserve.

What we KEEP intentionally:

- ``<system-reminder>`` and ``<local-command-*>`` user-as-message
  blocks. These are real framework events that influenced the
  conversation; filtering them would distort the as-of state CM
  reasons about. Adapters can be configured to drop them via
  ``include_system_reminders=False`` if the user prefers a cleaner
  archive at the cost of fidelity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import IngestedMessage, TranscriptAdapter

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCodeAdapter(TranscriptAdapter):
    """Adapter for ``~/.claude/projects/<slug>/<uuid>.jsonl`` files.

    Args:
        path: Filesystem path to the session JSONL.
        max_records: Optional cap on records read from the head of the
            file. Useful for testing on a partial transcript.
        include_system_reminders: If False, drop user-record content
            that begins with ``<system-reminder>`` or ``<local-command-``.
            Default True for fidelity.
    """

    path: Path
    max_records: Optional[int] = None
    include_system_reminders: bool = True
    name: str = "claude_code"

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def iter_messages(self) -> Iterator[IngestedMessage]:
        pending_tools: Dict[str, Dict[str, Any]] = {}
        turn_counter = 0

        for rec in _iter_records(self.path, self.max_records):
            rtype = rec.get("type")
            msg = rec.get("message") or {}
            content = msg.get("content")

            if rtype in ("attachment", "system"):
                continue

            if rtype == "user":
                turn_counter += 1
                yield from self._user_messages(
                    content, pending_tools, turn=turn_counter,
                    source_id=rec.get("uuid"),
                    source_metadata={"timestamp": rec.get("timestamp")},
                )
                continue

            if rtype == "assistant":
                yield from self._assistant_messages(
                    content, pending_tools, turn=turn_counter,
                    source_id=rec.get("uuid"),
                    source_metadata={"timestamp": rec.get("timestamp")},
                )
                continue

            logger.debug("claude_code: unknown record type %r — skipped", rtype)

        if pending_tools:
            logger.warning(
                "claude_code: %d tool_use blocks had no matching tool_result; "
                "they were dropped (sample: %s)",
                len(pending_tools), list(pending_tools)[:3],
            )

    # -- helpers -------------------------------------------------------

    def _user_messages(
        self,
        content: Any,
        pending_tools: Dict[str, Dict[str, Any]],
        *,
        turn: int,
        source_id: Optional[str],
        source_metadata: Dict[str, Any],
    ) -> Iterator[IngestedMessage]:
        if content is None:
            return

        if isinstance(content, str):
            if not self.include_system_reminders and content.startswith(
                ("<system-reminder>", "<local-command-")
            ):
                return
            if content.strip():
                yield IngestedMessage(
                    role="user", content=content, turn=turn,
                    source_id=source_id, source_metadata=source_metadata,
                )
            return

        if not isinstance(content, list):
            yield IngestedMessage(
                role="user", content=str(content), turn=turn,
                source_id=source_id, source_metadata=source_metadata,
            )
            return

        text_parts: List[str] = []
        # Image blocks within a user record (e.g., user pasted an
        # image into their message). Preserved as structured blocks
        # rather than flattened to "[image]" so vision-capable
        # downstream models can see the actual content.
        image_blocks: List[Dict[str, Any]] = []
        tool_results: List[IngestedMessage] = []

        for blk in content:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                t = blk.get("text") or ""
                if t:
                    text_parts.append(t)
            elif btype == "tool_result":
                tu_id = blk.get("tool_use_id") or ""
                tu = pending_tools.pop(tu_id, None)
                tool_name = (tu or {}).get("name") or "unknown"
                if tu is None:
                    # Orphan result — no matching tool_use was emitted
                    # earlier (truncated or malformed transcript).
                    # Synthesize a minimal assistant-with-tool_call so
                    # the live list still has a paired sender. The id
                    # is invented but unique within this run. Each
                    # orphan stays its own assistant message because
                    # we have no parent record to attach it to.
                    synth_id = tu_id or f"orphan_{turn}_{len(tool_results)}"
                    yield IngestedMessage(
                        role="assistant",
                        content="",
                        tool_calls=[{
                            "id": synth_id,
                            "type": "function",
                            "function": {"name": "unknown",
                                         "arguments": "{}"},
                        }],
                        turn=turn,
                        source_metadata={"orphan": True},
                    )
                    tu_id = synth_id
                # The non-orphan path no longer emits an assistant
                # here — ``_assistant_messages`` already emitted one
                # assistant message bundling all parallel tool_calls
                # from the parent record. Emitting per-result would
                # serialize parallel calls and lose the original
                # protocol-level structure.

                tool_results.append(IngestedMessage(
                    role="tool",
                    content=_extract_tool_result_text(blk.get("content")),
                    tool_call_id=tu_id,
                    name=tool_name,
                    turn=turn,
                    source_id=source_id,
                    source_metadata=source_metadata,
                ))
            elif btype == "image":
                # Preserve as structured block. Anthropic format:
                # {type: "image", source: {type: "base64"|"url", ...}}
                image_blocks.append(dict(blk))

        # Emit accumulated tool_results.
        for tr in tool_results:
            yield tr

        # Then any user text + image attachments on this record.
        if text_parts or image_blocks:
            joined_text = "\n".join(text_parts)
            if not self.include_system_reminders and joined_text.startswith(
                ("<system-reminder>", "<local-command-")
            ):
                return
            if image_blocks:
                # Mixed content: emit as a list of blocks so vision
                # models can see images alongside text.
                blocks: List[Dict[str, Any]] = []
                if joined_text:
                    blocks.append({"type": "text", "text": joined_text})
                blocks.extend(image_blocks)
                yield IngestedMessage(
                    role="user", content=blocks, turn=turn,
                    source_id=source_id, source_metadata=source_metadata,
                )
            else:
                yield IngestedMessage(
                    role="user", content=joined_text, turn=turn,
                    source_id=source_id, source_metadata=source_metadata,
                )

    def _assistant_messages(
        self,
        content: Any,
        pending_tools: Dict[str, Dict[str, Any]],
        *,
        turn: int,
        source_id: Optional[str],
        source_metadata: Dict[str, Any],
    ) -> Iterator[IngestedMessage]:
        if content is None:
            return

        if isinstance(content, str):
            if content.strip():
                yield IngestedMessage(
                    role="assistant", content=content, turn=turn,
                    source_id=source_id, source_metadata=source_metadata,
                )
            return

        if not isinstance(content, list):
            yield IngestedMessage(
                role="assistant", content=str(content), turn=turn,
                source_id=source_id, source_metadata=source_metadata,
            )
            return

        text_parts: List[str] = []
        image_blocks: List[Dict[str, Any]] = []
        # Thinking blocks (regular + redacted). Hermes preserves these
        # in ``reasoning_details`` so the Anthropic API's signature
        # validation works on round-trip. Dropping them would change
        # the conversation's KV state when the message is replayed.
        reasoning_blocks: List[Dict[str, Any]] = []
        # All parallel tool_uses from this assistant record bundled
        # onto a single assistant message. Splitting them across N
        # synthesized assistants on the user-record side would lose
        # the original protocol-level "one assistant turn, N parallel
        # calls" structure — visible to any replay or summarizer that
        # cares about call adjacency.
        tool_calls: List[Dict[str, Any]] = []

        for blk in content:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                t = blk.get("text") or ""
                if t:
                    text_parts.append(t)
            elif btype == "tool_use":
                tu_id = blk.get("id") or ""
                if tu_id:
                    tool_input = blk.get("input") or {}
                    tool_name = blk.get("name") or "unknown"
                    tool_calls.append({
                        "id": tu_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input,
                                                    ensure_ascii=False),
                        },
                    })
                    # Track the tool_use so the user-record handler
                    # can resolve orphans and so we can warn about
                    # tool_uses with no matching result at end-of-stream.
                    pending_tools[tu_id] = {
                        "name": tool_name,
                        "input": tool_input,
                        "source_id": source_id,
                    }
            elif btype in ("thinking", "redacted_thinking"):
                # Keep the whole block dict — signature, type, and all.
                reasoning_blocks.append(dict(blk))
            elif btype == "image":
                # Rare on the assistant side, but preserve it the same
                # way as on the user side.
                image_blocks.append(dict(blk))

        # Emit one assistant message per record, bundling text +
        # reasoning + tool_calls + images. The matching tool_results
        # surface on the NEXT user record per Claude Code's structure.
        # If only reasoning_blocks exist (no narration/calls), we
        # STILL emit so the signature is archived; otherwise it would
        # be lost across the replay.
        if text_parts or reasoning_blocks or image_blocks or tool_calls:
            if image_blocks:
                blocks: List[Dict[str, Any]] = []
                joined_text = "\n".join(text_parts)
                if joined_text:
                    blocks.append({"type": "text", "text": joined_text})
                blocks.extend(image_blocks)
                msg_content: Any = blocks
            else:
                msg_content = "\n".join(text_parts)
            yield IngestedMessage(
                role="assistant",
                content=msg_content,
                reasoning_details=reasoning_blocks or None,
                tool_calls=tool_calls or None,
                turn=turn,
                source_id=source_id,
                source_metadata=source_metadata,
            )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _iter_records(path: Path, max_records: Optional[int]):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("claude_code: malformed line %d skipped: %s", i, e)


def _extract_tool_result_text(content: Any) -> Any:
    """Tool result content can be string or list of typed blocks.

    Returns:
    - str if the result is plain text or all blocks are text-only
    - list of blocks if the result contains images (preserve structure
      so vision-capable consumers can use them)
    - "" for None
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        has_image = False
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") == "image":
                    has_image = True
                    break
        if has_image:
            # Return the structured list as-is. The autobio archive
            # JSON-encodes lists, so structure round-trips.
            return [b for b in content if isinstance(b, dict)]
        for blk in content:
            if isinstance(blk, dict):
                if blk.get("type") == "text":
                    text_parts.append(blk.get("text") or "")
            else:
                text_parts.append(str(blk))
        return "\n".join(text_parts)
    return str(content) if content is not None else ""

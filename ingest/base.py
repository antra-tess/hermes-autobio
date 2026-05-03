"""Adapter ABC and canonical message record.

Every source format (Claude Code JSONL, ChatGPT export, OpenAI Threads,
homegrown logs) produces an ``IngestedMessage`` stream in chronological
order. Downstream — ``seed_archive`` and the test harness alike — only
sees the canonical form, never source-specific structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class IngestedMessage:
    """One conversation message in canonical form.

    Mirrors the OpenAI chat-completions message shape plus the fields
    Hermes uses to preserve provider-specific state across summaries
    and round-trips:

    Standard fields:
    - ``role`` — "system" | "user" | "assistant" | "tool"
    - ``content`` — str OR list of content blocks (text/image/etc.).
      Structured content is preserved through to the archive (the
      autobio store JSON-encodes lists). Use a list when the message
      contains image attachments or other non-text parts.
    - ``tool_call_id`` — set when role == "tool"
    - ``tool_calls`` — set when an assistant emits tool calls
    - ``name`` — tool name on a tool message

    Hermes/provider-specific fields:
    - ``reasoning_details`` — list of thinking / redacted_thinking
      blocks Anthropic returned with the assistant message. Hermes'
      ``anthropic_adapter`` reads this back when the message is sent
      to Claude again, so signatures stay valid. We preserve it here
      so a transcript replay produces the same KV state the original
      session had.

    Ingestion bookkeeping:
    - ``turn`` — the source's notion of which turn this belongs to.
      For Claude Code, the user's request is one turn and all the
      subsequent assistant + tool messages inherit it. Used as the
      archive's per-row ``turn`` value, which feeds
      ``store.get_summary_turn_range`` for display labels.
    - ``source_id`` — the source's stable identifier for this message
      (e.g. Claude Code's ``uuid``). Informational only today.
    - ``source_metadata`` — adapter-specific hints (timestamp,
      sidechain flag, parent uuid, model name, usage). The seeder
      ignores it; downstream tools may surface it.
    """

    role: str
    content: Any
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None
    reasoning_details: Optional[List[Dict[str, Any]]] = None
    turn: int = 0
    source_id: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        """Render as a plain OpenAI-format message dict.

        Carries through provider-preserved fields (``reasoning_details``)
        because Hermes' adapter layer reads them on the way out. Drops
        ingestion-only fields (turn, source_id, source_metadata).
        """
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.name:
            out["name"] = self.name
        if self.reasoning_details:
            out["reasoning_details"] = self.reasoning_details
        return out


class TranscriptAdapter(ABC):
    """Abstract base for source-specific transcript loaders.

    Each adapter knows how to read its own format and yield
    ``IngestedMessage`` records in chronological order. Adapters do
    NOT touch the archive — they only produce canonical records. The
    seeder writes them.

    Implementations should be deterministic (same source → same
    stream of messages) and tolerate malformed input by logging and
    skipping rather than raising, so partial transcripts still
    migrate something rather than nothing.
    """

    #: Short identifier (e.g. "claude_code", "chatgpt_export"). Used in
    #: logs and CLI selection.
    name: str = "abstract"

    @abstractmethod
    def iter_messages(self) -> Iterator[IngestedMessage]:
        """Yield messages in chronological order. Must be re-runnable."""
        raise NotImplementedError

    def materialize(self) -> List[IngestedMessage]:
        """Drain the iterator into a list. Convenience for callers that
        want the full transcript at once (e.g., for stat reporting)."""
        return list(self.iter_messages())

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """Drain and convert to OpenAI-format message dicts."""
        return [m.to_openai() for m in self.iter_messages()]

"""Arc canonical-request-log adapter.

Arc can export a JSON file containing the exact request that was sent
to the LLM API for the most recent turn — system prompt, alternating
user/assistant messages with multi-participant turns already
serialized as ``Name: content`` prefixes (Arc's prefill-format render),
stop sequences, and provider settings.

Why use the request log instead of the conversation export:

  - The conversation JSON is a tree (multiple branches per message,
    consecutive same-role messages, multi-participant turns); rendering
    it back into API shape requires reproducing Arc's formatter.
  - The request log IS the API shape. Every transformation Arc applied
    is already baked in: prefill→messages conversion, multi-participant
    serialization with ``Name:`` prefixes, consecutive-same-role
    collapsing.
  - For a transfer that aims to minimize KV disturbance, ingesting the
    exact bytes that conditioned the agent's most recent KV state is
    the highest-fidelity path available.

Top-level shape of a canonical request log:

    {
      "model": "claude-opus-4-6",
      "system": "...",
      "messages": [
        {"role": "user", "content": "Antra: ..."},
        {"role": "assistant", "content": "..."},
        ...
      ],
      "max_tokens": 8096,
      "temperature": 1,
      "stop_sequences": [...],
      ...
    }

We ingest the ``messages`` list. Other fields (``system``,
``stop_sequences``, ``max_tokens``, ``temperature``, ``thinking``)
are session-config concerns the operator wires into Hermes separately;
they do not become archive messages. ``system`` notably stays out of
the archive — Anthropic's API takes it as a top-level field, not as
``messages[0]``.

The optional ``final_assistant_text`` argument appends one synthetic
assistant turn after the request log's last message. Use when the
request log captured only the *input* that produced an assistant
reply, and you want the archive to include that reply as the most
recent turn (so the live view starts where the agent's KV state
actually was, not one turn earlier).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import IngestedMessage, TranscriptAdapter

logger = logging.getLogger(__name__)


@dataclass
class ArcRequestLogAdapter(TranscriptAdapter):
    """Adapter for an Arc ``canonical_request_log.json`` file.

    Args:
        path: Filesystem path to the request log JSON.
        final_assistant_text: Optional. If given, appended as one more
            assistant turn after the request log's last message. Use
            when the log captured the prompt that produced a reply
            and you want the archive to end with that reply.
        max_records: Optional cap on records read from the head of
            the messages list. Useful for partial-import testing.
    """

    path: Path
    final_assistant_text: Optional[str] = None
    max_records: Optional[int] = None
    name: str = "arc_request_log"

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def iter_messages(self) -> Iterator[IngestedMessage]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        msgs = data.get("messages")
        if not isinstance(msgs, list):
            logger.error(
                "arc_request_log: %s has no ``messages`` list at top level",
                self.path,
            )
            return

        model = data.get("model")
        turn = 0
        emitted = 0
        last_role: Optional[str] = None
        for i, entry in enumerate(msgs):
            if self.max_records is not None and emitted >= self.max_records:
                break
            if not isinstance(entry, dict):
                logger.debug(
                    "arc_request_log: skipping non-dict entry at index %d", i,
                )
                continue
            role = entry.get("role")
            content = entry.get("content")
            if role not in ("user", "assistant"):
                logger.debug(
                    "arc_request_log: skipping role=%r at index %d", role, i,
                )
                continue
            if content is None:
                continue
            # Coerce non-string content to string (the request log is
            # uniformly stringy in practice, but we don't trust the
            # source unconditionally).
            if not isinstance(content, str):
                content = str(content)

            # Turn counter: a user message starts a new turn; an
            # assistant message inherits the most recent turn. Mirrors
            # the convention used by the Claude Code adapter so the
            # archive's per-row ``turn`` value is comparable across
            # ingest sources.
            if role == "user":
                turn += 1

            metadata: Dict[str, Any] = {
                "source": "arc_request_log",
                "request_log_index": i,
            }
            if model:
                metadata["model"] = model

            yield IngestedMessage(
                role=role,
                content=content,
                turn=turn,
                source_metadata=metadata,
            )
            emitted += 1
            last_role = role

        if self.final_assistant_text:
            # Synthetic cap: the assistant reply produced FROM the
            # request log. Inherit the latest turn — this reply
            # belongs to the same turn as the user message that
            # prompted it. Tagged synthetic_cap so a future audit
            # can distinguish it from a request-log-sourced row.
            if last_role == "assistant":
                logger.warning(
                    "arc_request_log: appending final_assistant_text after "
                    "an assistant message (request log already ended on "
                    "assistant). The Anthropic messages API rejects "
                    "consecutive same-role turns; verify intent.",
                )
            yield IngestedMessage(
                role="assistant",
                content=self.final_assistant_text,
                turn=turn,
                source_metadata={
                    "source": "arc_request_log",
                    "synthetic_cap": True,
                },
            )

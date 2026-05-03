"""Transcript ingest for autobio.

Loads conversation history from external sources (Claude Code session
JSONL today; ChatGPT export, custom logs, etc. in the future) and
seeds it into an autobio session's archive. The intended use is
migrating a long-lived instance off some other harness onto Hermes
without losing its existing context.

Public API:

- ``TranscriptAdapter`` — abstract base for source-specific loaders.
- ``IngestedMessage`` — canonical record yielded by adapters.
- ``ClaudeCodeAdapter`` — concrete adapter for Claude Code JSONL.
- ``seed_archive(store, adapter, ...)`` — write an adapter's stream
  into the archive in chronological order.
"""

from __future__ import annotations

from .base import IngestedMessage, TranscriptAdapter
from .claude_code import ClaudeCodeAdapter
from .redact import redact_message, redact_text
from .seed import SeedResult, seed_archive

__all__ = [
    "IngestedMessage",
    "TranscriptAdapter",
    "ClaudeCodeAdapter",
    "seed_archive",
    "SeedResult",
    "redact_text",
    "redact_message",
]

"""Autobio viewer — Flask debugging UI for live + archived sessions.

Reads from the autobio store and renders:

- Session picker (root)
- Per-session three-pane view: chronological messages, memory hierarchy
  (L3→L2→L1→messages), and detail panel for the currently selected node.
- For each summary: the exact LLM prompt that produced it, rendered as
  a turn-by-turn conversation.
- For each CM event: request, dashboard, the prompt fed to CM, raw
  reply, decision, ops + their results.

CLI:

    python -m plugins.autobio.viewer --hermes-home ~/.hermes [--port 8765]
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]

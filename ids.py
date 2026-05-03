"""Content-hash message IDs.

Messages are identified by a SHA-256 hash of their semantic content
(role + body + tool metadata), truncated to 16 hex chars. The hash is
minted once when the message enters the archive and is stable across
list-index shifts, pruning, and compression.

Rationale — see the design spec discussion on hashes vs. minted IDs:
hashes survive renumbering and auto-dedup, but require care about what
goes in. We hash the *archival* form (full content, immutable), never
the pruned/live-list form, so tool-result pruning does not invalidate
identifiers downstream.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


_HASH_LEN = 16  # 64 bits; collision risk negligible at realistic scale


def message_hash(message: Dict[str, Any]) -> str:
    """Return the stable content-hash ID for a single message dict.

    The hash input is a canonicalized JSON of (role, content, tool_call_id,
    tool_calls). Assistant messages with tool_calls include their call IDs
    and function names + args so two otherwise-identical assistant
    messages with different tool calls produce distinct hashes.
    """
    payload = {
        "role": message.get("role", ""),
        "content": _normalize_content(message.get("content")),
        "tool_call_id": message.get("tool_call_id", ""),
        "tool_name": message.get("name", ""),
        "tool_calls": _normalize_tool_calls(message.get("tool_calls")),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_HASH_LEN]


def combined_hash(hashes: List[str]) -> str:
    """Hash a list of message/summary hashes into a single combined hash.

    Used by the speculative compression worker to identify a full
    compression input (preceding summaries + tail + chunk). Order matters —
    a rollback that removes a message produces a different combined hash
    and invalidates the cached speculative result.
    """
    blob = "|".join(hashes).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


def _normalize_content(content: Any) -> Any:
    """Normalize message content for hashing.

    Strings pass through. Content lists (multimodal) are sorted-key JSON.
    Anything else is coerced to str.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_normalize_content_block(b) for b in content]
    return str(content)


def _normalize_content_block(block: Any) -> Any:
    if isinstance(block, dict):
        return {k: block[k] for k in sorted(block.keys())}
    return str(block)


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize assistant tool_calls for hashing.

    Handles both dict-form (OpenAI JSON) and SimpleNamespace-form
    (ChatCompletion objects from the SDK).
    """
    if not tool_calls:
        return []
    out = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            out.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            })
        else:
            fn = getattr(tc, "function", None)
            out.append({
                "id": getattr(tc, "id", "") or "",
                "name": (getattr(fn, "name", "") or "") if fn else "",
                "arguments": (getattr(fn, "arguments", "") or "") if fn else "",
            })
    return out

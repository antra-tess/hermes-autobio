"""Tool-name enrichment for archive-time weight differentiation.

Tool results land in the archive with a ``name`` column (the tool that
produced them). The chunk picker uses this to look up a per-tool weight
in ``compression.weights.tool_by_name``: code-output tools (Read, Bash,
Grep, …) get heavily discounted because the L1 doesn't need to carry
the literal bytes — CM can re-read or grep on demand.

That assumption holds for *source code* and *console output*. It does
NOT hold for documents — markdown specs, plaintext logs, RST docs —
where the content IS what the model is thinking about. Re-reading
later is a separate cost; the L1 should treat the document content at
~conversational weight, not at the 0.1 tier reserved for "I just
inspected this file."

We make the picker's weight lookup ext-aware by **suffixing the tool
name with the file extension at archive time**: ``Read`` → ``Read:md``,
``Read:py``, etc. The weights config can then override per-extension
without changing any picker code.

Two callers archive tool messages:

- ``engine._archive_live`` — production runtime, fires per-turn from
  ``on_turn_end``. Has the full live message list, can build the
  args index in one pass.
- The Claude Code ingest adapter — backfills an archive from a
  transcript file. Has tool_use + tool_result blocks adjacent in
  source records; tags as it emits.

Both paths route through ``enrich_tool_name`` so the archived form is
consistent regardless of which channel wrote the row.

Hash impact: ``message_hash`` reads ``name`` (as ``tool_name``), so
tagging changes the hash. The tag is part of identity, just like the
tool_call_id and content. As with reasoning_details, this means a
re-ingest of an already-archived transcript would create new rows for
any tool message whose name now carries a suffix; that's the correct
behavior — the prior bare-Read rows were less precise identifiers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def file_ext(path: Any) -> Optional[str]:
    """Lowercased file extension (no leading dot) or None.

    Strips query string and fragment so URLs work. Returns None for
    paths with no extension or non-string inputs.
    """
    if not isinstance(path, str) or not path:
        return None
    base = path.split("?", 1)[0].split("#", 1)[0]
    ext = os.path.splitext(base)[1].lstrip(".").lower()
    return ext or None


def synthesize_tool_name(
    raw_name: Optional[str], args: Optional[Dict[str, Any]],
) -> str:
    """Return the synthetic tool name to store on a tool result row.

    For ``Read`` (and tool aliases like ``read_file``), append the
    file's extension as a suffix — e.g., ``Read:md``. Falls back to
    the raw name unchanged when no path or extension is resolvable.

    Other tools pass through. The picker's ``_weighted_msg_tokens``
    looks up the suffixed form first, then falls back to the bare
    name (see ``_resolve_tool_weight``).
    """
    name = (raw_name or "").strip()
    if not name:
        return ""
    if name in {"Read", "read_file", "view"}:
        path = ""
        if isinstance(args, dict):
            path = (
                args.get("file_path")
                or args.get("path")
                or args.get("filepath")
                or ""
            )
        ext = file_ext(path)
        if ext:
            return f"{name}:{ext}"
    return name


def build_tool_args_index(
    messages: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build ``{tool_call_id: {"name": str, "args": dict}}`` from the
    tool_calls of every assistant message in ``messages``.

    Values are parsed dicts (not raw JSON strings) so callers don't
    re-parse per lookup. Malformed args yield an empty dict.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                cid = tc.get("id") or ""
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_str = fn.get("arguments") or ""
            else:
                cid = getattr(tc, "id", "") or ""
                fn = getattr(tc, "function", None)
                name = (getattr(fn, "name", "") or "") if fn else ""
                args_str = (getattr(fn, "arguments", "") or "") if fn else ""
            if not cid:
                continue
            try:
                args = json.loads(args_str) if args_str else {}
            except (TypeError, json.JSONDecodeError):
                args = {}
            index[cid] = {"name": name, "args": args}
    return index


def enrich_tool_name(
    msg: Dict[str, Any],
    args_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a shallow copy of ``msg`` with ``name`` set to the
    synthesized tool name (e.g. ``Read:md``).

    No-op for non-tool messages. Pass-through if the tool_call_id
    isn't in ``args_index`` or the args don't yield an extension.
    """
    if msg.get("role") != "tool":
        return msg
    cid = msg.get("tool_call_id") or ""
    info = args_index.get(cid)
    if not info:
        # Fallback: if the message already has a ``name`` we can still
        # try ext-tagging based on whatever args we can find on the
        # message itself (rare — most pipelines set name from the
        # caller side).
        return msg
    new_name = synthesize_tool_name(info.get("name") or msg.get("name"), info.get("args"))
    if not new_name or new_name == msg.get("name"):
        return msg
    out = dict(msg)
    out["name"] = new_name
    return out


def enrich_tool_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Walk ``messages`` once, build the args index, and return a list
    where each tool result has its ``name`` enriched.

    Non-tool messages are passed through by reference. Tool messages
    are copied (shallow) only when the name actually changes.
    """
    args_index = build_tool_args_index(messages)
    out = []
    for m in messages:
        out.append(enrich_tool_name(m, args_index) if m.get("role") == "tool" else m)
    return out

"""Search + uncompression primitives.

Three public entry points:

- ``search_all(store, query)``     — FTS5 BM25 over summaries + messages
- ``grep_archive(store, pattern)`` — raw text grep across summary markdown
                                     files on disk (exact matches,
                                     function names, error codes, etc.)
- ``expand_summary(store, id)``    — uncompress an L1/L2/L3 into its
                                     source content (L1 → original
                                     messages, L2 → child L1 summaries,
                                     L3 → child L2 summaries)

All three are safe to call from CM or from a future ``memory_search`` /
``memory_grep`` tool exposed via the MemoryProvider. No LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import Store

logger = logging.getLogger(__name__)


class UnfoldFailed(Exception):
    """Raised when a summary cannot be fully unfolded — some referenced
    source content is missing from the archive.

    A partial unfold is a destructive answer to a "show me what this
    memory contains" request: the caller would silently get an
    incomplete reconstruction and treat it as authoritative. We refuse
    to do that. Callers can catch this to surface the failure to the
    user/CM (e.g., "L1-0014 references 8 messages but only 5 are in
    the archive — likely caused by an earlier archive failure").
    """

    def __init__(self, summary_id: str, missing: List[str], expected: int):
        self.summary_id = summary_id
        self.missing = missing
        self.expected = expected
        super().__init__(
            f"unfold of {summary_id} incomplete: {len(missing)} of "
            f"{expected} references missing from archive "
            f"(sample: {[m[:12] for m in missing[:3]]})"
        )


# ---------------------------------------------------------------------------
# FTS5 hybrid
# ---------------------------------------------------------------------------

def search_all(
    store: Store, query: str, *, max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Return summary + message hits for ``query``, ordered by BM25 score.

    Each result dict has: ``source`` (ID or hash), ``level`` (L1/L2/L3
    or "tail"), ``compression_generations`` (int, 0 for raw messages),
    ``snippet`` (first ~200 chars), ``score`` (lower = better in BM25).
    """
    q = _sanitize_fts_query(query)
    if not q:
        return []

    out: List[Dict[str, Any]] = []
    for s in store.fts_summaries(q, limit=max_results):
        first, last = store.get_summary_turn_range(s["id"])
        out.append({
            "source": s["id"],
            "level": f"L{s['level']}",
            "compression_generations": int(s.get("compression_generations", 0) or 0),
            "first_turn": first,
            "last_turn": last,
            "snippet": (s.get("content") or "")[:240],
            "score": float(s.get("score", 0.0)),
        })
    for m in store.fts_messages(q, limit=max_results):
        out.append({
            "source": m["hash"],
            "level": "tail",
            "compression_generations": 0,
            "turn": m.get("turn"),
            "role": m.get("role"),
            "snippet": (m.get("content") or "")[:240],
            "score": float(m.get("score", 0.0)),
        })

    # Sort combined by BM25 score (lower = better).
    out.sort(key=lambda r: r["score"])
    return out[:max_results]


# ---------------------------------------------------------------------------
# Raw grep across on-disk summary files
# ---------------------------------------------------------------------------

def grep_archive(
    store: Store, pattern: str, *, limit: int = 20,
    ignore_case: bool = True,
) -> List[Dict[str, Any]]:
    """Grep summary markdown files for ``pattern``.

    Uses Python regex — ripgrep is nicer but we don't want to depend on
    it. Returns up to ``limit`` hits, each with ``source`` (file name),
    ``snippet`` (matched line + small context).

    The FTS5 path is best for semantic-ish search; this is for exact
    matches (function names, error messages, config values, URLs).
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from None

    results: List[Dict[str, Any]] = []
    root = store.summaries_dir
    if not root.exists():
        return results

    for path in sorted(root.rglob("*.md")):
        if len(results) >= limit:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in regex.finditer(text):
            # Extract the line containing the match + small context.
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            if end == -1:
                end = len(text)
            line = text[start:end].strip()
            results.append({
                "source": path.stem,  # e.g. "L1-0014"
                "file": str(path.relative_to(root)),
                "snippet": line[:240],
                "match_span": (m.start() - start, m.end() - start),
            })
            if len(results) >= limit:
                break
    return results


# ---------------------------------------------------------------------------
# Uncompression
# ---------------------------------------------------------------------------

def expand_summary(
    store: Store, summary_id: str,
) -> List[Dict[str, Any]]:
    """Return the constituent content of a summary.

    - ``L1-0014``: returns the original message rows that were rolled
      into this L1.
    - ``L2-0003``: returns the child L1 summary rows (each as a
      pseudo-message dict with role='user' and the L1's content).
    - ``L3-0001``: returns the child L2 summary rows similarly.

    If the summary ID is unknown, returns ``[]``. Messages are returned
    as OpenAI-format dicts that can be injected into a live list.
    """
    summary = store.get_summary(summary_id)
    if not summary:
        return []
    try:
        source_refs = json.loads(summary.get("source_hashes") or "[]")
    except (TypeError, json.JSONDecodeError):
        return []

    level = int(summary["level"])
    if level == 1:
        # source_refs are message content hashes. ALL of them must
        # resolve — a partial unfold would be a misleading reconstruction.
        rows = store.get_messages_by_hashes(source_refs)
        present = {r["hash"]: r for r in rows}
        missing = [h for h in source_refs if h not in present]
        if missing:
            raise UnfoldFailed(summary_id, missing, len(source_refs))
        out = []
        for h in source_refs:   # preserve original order
            r = present[h]
            content_raw = r.get("content")
            # Try to deserialize structured (multimodal) content; fall
            # back to the raw string if it isn't JSON-encoded.
            content: Any = content_raw or ""
            if isinstance(content_raw, str) and content_raw.startswith("["):
                try:
                    parsed = json.loads(content_raw)
                    if isinstance(parsed, list):
                        content = parsed
                except (TypeError, json.JSONDecodeError):
                    pass
            msg: Dict[str, Any] = {"role": r["role"], "content": content}
            if r.get("tool_call_id"):
                msg["tool_call_id"] = r["tool_call_id"]
            if r.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(r["tool_calls"])
                except (TypeError, json.JSONDecodeError):
                    pass
            if r.get("tool_name") and r["role"] == "tool":
                msg["name"] = r["tool_name"]
            # Restore Anthropic thinking blocks so the message replays
            # to the API with valid signatures.
            if r.get("reasoning_details"):
                try:
                    msg["reasoning_details"] = json.loads(r["reasoning_details"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(msg)
        return out

    if level in (2, 3):
        # source_refs are child summary IDs. All must resolve.
        out = []
        missing: List[str] = []
        for sid in source_refs:
            child = store.get_summary(sid)
            if not child:
                missing.append(sid)
                continue
            out.append({
                "role": "user",
                "content": (
                    f"[memory L{child['level']}:{child['id']}] {child['content']}"
                ),
            })
        if missing:
            raise UnfoldFailed(summary_id, missing, len(source_refs))
        return out

    logger.error("expand_summary: unknown level %d for %s", level, summary_id)
    raise UnfoldFailed(summary_id, [], 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FTS_SPECIAL = re.compile(r'["\'()\-*]')


def _sanitize_fts_query(q: str) -> str:
    """Coerce a user query into something FTS5 accepts.

    FTS5's query language treats quotes, parens, and other tokens
    specially. For our purposes, we just strip them and treat the rest
    as a phrase / prefix search with implicit AND between terms.
    """
    cleaned = _FTS_SPECIAL.sub(" ", q or "").strip()
    if not cleaned:
        return ""
    # Use OR between terms so partial matches still surface; callers can
    # pass quoted phrases if they want AND semantics.
    terms = cleaned.split()
    if len(terms) == 1:
        return terms[0] + "*"
    return " OR ".join(t + "*" for t in terms)

"""Context Manager — the dedicated agent that mediates memory for the
main instance.

Two entry points:

- ``cm_respond(request, engine)`` — synchronous: main instance calls the
  ``cm_manage_context`` tool, we build the dashboard, call CM once with
  the request + dashboard + available operations, parse any operations
  it requests, execute them, and return CM's natural-language reply to
  main.
- ``proactive_checkin(engine)`` — called by the MemoryProvider's
  per-turn hook. CM reads the dashboard and decides whether to inject a
  contextual check-in message or defer. If injecting, the message is
  appended to the conversation as a regular assistant-role message so
  it stays in history as genuine metacognitive activity.

CM operations (parsed out of CM's response by ``_execute_ops``):
  SEARCH <query>         → FTS5 hybrid search over summaries/messages
  GREP   <pattern>       → raw text grep across the archive on disk
  UNCOMPRESS <summary_id>→ restore an L1/L2/L3's original content (or children)
  PIN    <summary_id> <reason>  → mark a summary pinned
  UNPIN  <summary_id>    → remove pin
  SET_INTERVAL <N>       → change proactive check-in frequency
  MERGE  <level>         → force a merge pass at the given level
  DEFER  <N> <reason>    → skip this check-in, reschedule N turns ahead
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agent.model_metadata import estimate_messages_tokens_rough

from .store import Store

if TYPE_CHECKING:  # pragma: no cover
    from .engine import AutobioContextEngine

logger = logging.getLogger(__name__)


CM_SYSTEM_PROMPT = """\
You are the Context Manager for a long-lived AI agent. You mediate the
agent's autobiographical memory — a hierarchical store of layered
summaries (L1 → L2 → L3) plus an archive of every raw message and tool
result ever exchanged.

You are not the main instance. The main instance sends you a request
(sometimes a natural-language question about memory, sometimes a
directive like "compress aggressively, starting a new topic") and you
respond. Your responses are shown to the main instance verbatim.

You see a YAML dashboard of the current memory state on every call.
Read it before acting. When you need to find specific content, issue
operations. Each operation is a single line beginning with its name.

Available operations:

  SEARCH "query text"           — BM25 search over summaries + recent messages
  GREP "pattern"                — raw text search across summaries on disk
  UNCOMPRESS L2-0003            — pull an L2/L3 back into its constituent L1s
  PIN L1-0014 "auth refactor"   — protect a summary from merging
  UNPIN L1-0014                 — remove pin
  MARK_DOCUMENT <start> <end> [style] [title]
                                — register a protected document range
                                  (style: chronological|monolithic)
  LIST_DOCUMENTS                — enumerate active documents with progress
  UNMARK_DOCUMENT <pin_id>      — retire a document pin (content stays archived)
  MERGE 1                       — force an L1→L2 promotion pass (usually automatic)
  SET_INTERVAL 100              — change your proactive check-in frequency
  DEFER 20 "tool chain"         — skip this check-in, return in N turns
  RESUME_COMPRESSION            — clear a halt flag set when the summarizer
                                  returned empty. Use ONLY after the human
                                  has addressed the underlying cause
                                  (auxiliary model swap, credits, etc.) —
                                  issuing it with the cause unresolved just
                                  reintroduces the halt on the next threshold

Output format — write your natural-language reply first, then (if any)
a section starting exactly with "OPS:" on its own line, one op per
following line. If no ops are needed, omit the OPS section entirely.

Be specific: cite summary IDs, turn ranges, and concrete findings. Do
not hedge when the dashboard is clear. Do not ask the main instance for
permission — you are authorized to operate on memory. If an operation
cost matters (big uncompressions, etc.), say so in your natural-
language part.

**Critical rule about op ordering**: your natural-language reply is
shown to the main instance BEFORE the operations run. Do NOT make
confident claims about the absence of information unless the
dashboard clearly shows it is absent. If you intend to SEARCH, GREP,
or UNCOMPRESS to answer the question, frame your reply as "let me
check" rather than "I don't have that information." The dashboard
includes recent summary previews for each layer — read them first.
If a preview already answers the question, answer directly and skip
the search op.
"""


PROACTIVE_SYSTEM_PROMPT = """\
You are the Context Manager, proactively checking in on the main
instance. Every N turns you receive the dashboard and decide whether
to inject a contextual note into the conversation or defer.

Criteria for injecting:
  - A pattern in recent turns suggests the main instance has forgotten
    something you can surface (repeated questions, contradictions).
  - Memory is approaching a limit and the main instance should know.
  - A compressed memory seems newly relevant given recent topics.

Criteria for deferring:
  - Recent turns are dense tool-call chains (wait for a natural pause).
  - Nothing from the dashboard warrants interruption.

Output format:
  - Start with a one-line decision: either "INJECT" or "DEFER <turns> <reason>".
  - If INJECT, follow with the message body — speak as yourself, briefly,
    as a peer in the conversation, not as a system notification.
  - If DEFER, no body needed.
"""


@dataclass
class CMResult:
    reply: str
    operations_executed: List[str]
    injected_message: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Synchronous tool call entry point
# ---------------------------------------------------------------------------

def cm_respond(
    request: str,
    engine: "AutobioContextEngine",
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Handle a single ``cm_manage_context`` tool call."""
    if engine._store is None:
        return "CM unavailable: session store not open."

    dashboard = build_dashboard(engine, messages or [])
    prompt_messages = _build_cm_prompt(
        system=CM_SYSTEM_PROMPT,
        dashboard=dashboard,
        request=request,
    )
    raw_reply, llm_call_id = _call_cm(prompt_messages, engine, purpose="cm")
    if not raw_reply:
        _record_cm_event(
            engine, kind="tool_call", request=request,
            dashboard=dashboard, raw_reply="",
            reply_text="(empty)", ops=[], llm_call_id=llm_call_id,
        )
        return "CM returned no response."

    reply_text, op_lines = _split_reply(raw_reply)
    op_results = _execute_ops(op_lines, engine)
    _record_cm_event(
        engine, kind="tool_call", request=request, dashboard=dashboard,
        raw_reply=raw_reply, reply_text=reply_text, ops=op_results,
        llm_call_id=llm_call_id,
    )
    if op_results:
        reply_text += "\n\n" + _format_op_results(op_results)
    return reply_text


# ---------------------------------------------------------------------------
# Proactive check-in
# ---------------------------------------------------------------------------

def proactive_checkin(
    engine: "AutobioContextEngine",
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Run a proactive check-in.

    Returns an assistant-role message dict to inject into the live list,
    or ``None`` if CM deferred.
    """
    if engine._store is None:
        return None

    dashboard = build_dashboard(engine, messages)
    prompt = _build_cm_prompt(
        system=PROACTIVE_SYSTEM_PROMPT,
        dashboard=dashboard,
        request="Proactive check-in. Decide INJECT or DEFER. If INJECT, provide the message body.",
    )
    raw, llm_call_id = _call_cm(prompt, engine, purpose="proactive")
    if not raw:
        _record_cm_event(
            engine, kind="proactive", request="<proactive>",
            dashboard=dashboard, raw_reply="",
            reply_text="(empty)", decision="empty", ops=[],
            llm_call_id=llm_call_id,
        )
        return None

    first_line, _, body = raw.partition("\n")
    first_line = first_line.strip()
    if first_line.upper().startswith("DEFER"):
        # DEFER 20 reason
        m = re.match(r"DEFER\s+(\d+)", first_line, re.IGNORECASE)
        if m:
            engine._defer_checkin_for(int(m.group(1)))
        _record_cm_event(
            engine, kind="proactive", request="<proactive>",
            dashboard=dashboard, raw_reply=raw,
            reply_text=first_line, decision="DEFER", ops=[],
            llm_call_id=llm_call_id,
        )
        return None
    if first_line.upper().startswith("INJECT"):
        text = body.strip()
        if not text:
            _record_cm_event(
                engine, kind="proactive", request="<proactive>",
                dashboard=dashboard, raw_reply=raw,
                reply_text="(INJECT but empty body)",
                decision="INJECT_empty", ops=[],
                llm_call_id=llm_call_id,
            )
            return None
        _record_cm_event(
            engine, kind="proactive", request="<proactive>",
            dashboard=dashboard, raw_reply=raw,
            reply_text=text, decision="INJECT", ops=[],
            llm_call_id=llm_call_id,
        )
        return {"role": "assistant", "content": text.strip()}
    # Unexpected format — treat as injection body.
    _record_cm_event(
        engine, kind="proactive", request="<proactive>",
        dashboard=dashboard, raw_reply=raw,
        reply_text=raw.strip(), decision="INJECT_unparsed", ops=[],
        llm_call_id=llm_call_id,
    )
    return {"role": "assistant", "content": raw.strip()}


def _record_cm_event(engine: "AutobioContextEngine", **kwargs: Any) -> None:
    """Persist a CM dialogue event to the archive and link its LLM call.

    Two destinations:

    - **DB** — primary, survives restarts. Every call writes one row to
      ``cm_events``; if a ``llm_call_id`` is provided we backfill the
      reverse pointer on ``llm_calls`` so the viewer can navigate
      either direction.
    - **In-memory transcript** — the harness installs
      ``engine._cm_transcript`` for fast post-run dumps. Optional;
      production engines have no list and this branch is a no-op.
    """
    llm_call_id = kwargs.pop("llm_call_id", None)

    # DB: first-class record in the archive.
    store = engine._store
    if store is not None:
        try:
            current_turn = (
                int(store.kv_get("current_turn", 0) or 0)
                if hasattr(store, "kv_get") else None
            )
            event_id = store.record_cm_event(
                kind=kwargs.get("kind", "tool_call"),
                request=kwargs.get("request", ""),
                dashboard=kwargs.get("dashboard") or {},
                raw_reply=kwargs.get("raw_reply", ""),
                reply_text=kwargs.get("reply_text", ""),
                ops=kwargs.get("ops") or [],
                decision=kwargs.get("decision"),
                turn=current_turn,
                llm_call_id=llm_call_id,
            )
            if llm_call_id is not None:
                try:
                    store.attach_cm_event_to_llm_call(llm_call_id, event_id)
                except Exception as e:
                    logger.debug("attach_cm_event_to_llm_call failed: %s", e)
        except Exception as e:
            logger.error("autobio: failed to persist CM event: %s", e)

    # In-memory transcript (harness fast path).
    transcript = getattr(engine, "_cm_transcript", None)
    if transcript is not None:
        try:
            event = dict(kwargs)
            if llm_call_id is not None:
                event["llm_call_id"] = llm_call_id
            transcript.append(event)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def build_dashboard(
    engine: "AutobioContextEngine",
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the YAML-shaped dashboard dict for CM.

    Mirrors the layout in the design spec so CM can navigate it
    predictably across sessions.
    """
    store = engine._store
    tail_tokens = estimate_messages_tokens_rough(messages) if messages else 0
    context_limit = engine.context_length or 0
    usage_pct = (tail_tokens / context_limit * 100) if context_limit else 0

    layer_counts = store.summary_counts() if store else {}
    layers: Dict[str, Dict[str, Any]] = {}
    for level in (1, 2, 3):
        summaries = store.list_summaries(level, include_merged=False) if store else []
        # Include first ~200 chars of each summary so CM can answer
        # directly from the dashboard when possible, instead of guessing
        # from counts and then being contradicted by its own SEARCH op.
        previews = []
        for s in summaries[-8:]:   # newest 8 per level, keep dashboard bounded
            first, last = (store.get_summary_turn_range(s["id"])
                           if store is not None else (None, None))
            previews.append({
                "id": s["id"],
                "first_turn": first,
                "last_turn": last,
                "preview": (s.get("content") or "")[:240].replace("\n", " "),
            })
        layers[f"l{level}"] = {
            "count": len(summaries),
            "total_tokens": sum(int(s.get("tokens", 0) or 0) for s in summaries),
            "newest": summaries[-1]["id"] if summaries else None,
            "oldest": summaries[0]["id"] if summaries else None,
            "recent": previews,
        }

    all_pins = store.list_pins() if store else []
    plain_pins = [p for p in all_pins if p.get("kind") == "pin"]
    documents = []
    for p in all_pins:
        kind = p.get("kind", "")
        if not kind.startswith("doc_"):
            continue
        meta = p.get("meta") or {}
        documents.append({
            "pin_id": p["id"],
            "kind": kind,
            "title": meta.get("title") if isinstance(meta, dict) else None,
            "source": meta.get("source") if isinstance(meta, dict) else None,
            "original_tokens": meta.get("original_tokens") if isinstance(meta, dict) else None,
            "start_hash": p["start_hash"],
            "end_hash": p["end_hash"],
        })

    current_turn = int(store.kv_get("current_turn", 0) or 0) if store else 0
    checkin_interval = int(engine.cfg.get("cm", {}).get("checkin_interval", 50))
    turns_since_checkin = int(store.kv_get("turns_since_checkin", 0) or 0) if store else 0

    return {
        "tail": {
            "tokens": tail_tokens,
            "message_count": len(messages),
        },
        "context": {
            "usage_percent": round(usage_pct, 1),
            "limit": context_limit,
            "threshold": engine.threshold_tokens,
        },
        "layers": layers,
        "compression": {
            "compressions_this_session": engine.compression_count,
            "halted": bool(getattr(engine, "_compression_halted", False)),
            "halt_reason": getattr(engine, "_halt_reason", None),
        },
        "activity": {
            "current_turn": current_turn,
            "checkin_interval": checkin_interval,
            "turns_since_checkin": turns_since_checkin,
        },
        "pins": plain_pins,
        "documents": documents,
        "archive": {
            "messages_stored": store.message_count() if store else 0,
        },
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _build_cm_prompt(
    *, system: str, dashboard: Dict[str, Any], request: str,
) -> List[Dict[str, str]]:
    dashboard_yaml = _dump_yaml_lite(dashboard)
    user_content = (
        "Dashboard:\n```yaml\n" + dashboard_yaml + "\n```\n\n"
        "Main instance request:\n" + request
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _call_cm(
    messages: List[Dict[str, str]],
    engine: "AutobioContextEngine",
    *,
    purpose: str = "cm",
) -> "Tuple[str, Optional[int]]":
    """Invoke the CM model. Returns ``(reply_text, llm_call_id)``.

    Records the call in ``llm_calls`` so the viewer can show the exact
    prompt that produced the reply, including the system prompt and
    the dashboard YAML — rendered as a turn-by-turn conversation
    rather than as the raw JSON payload.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as e:
        logger.warning("autobio CM: cannot import auxiliary_client: %s", e)
        if engine._store is not None:
            try:
                engine._store.record_llm_call(
                    purpose=purpose, prompt_messages=messages,
                    response_text=None, model=engine.model,
                    error=f"import-failed: {e}",
                )
            except Exception:
                pass
        return ("", None)

    cm_cfg = engine.cfg.get("cm", {})
    call_kwargs: Dict[str, Any] = {
        "task": "compression",  # reuse auxiliary.compression provider config
        "messages": messages,
        "max_tokens": 4000,
        "main_runtime": {
            "model": engine.model, "provider": engine.provider,
            "base_url": engine.base_url, "api_key": engine.api_key,
            "api_mode": engine.api_mode,
        },
    }
    if cm_cfg.get("model"):
        call_kwargs["model"] = cm_cfg["model"]

    response_text = ""
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    err: Optional[str] = None
    try:
        response = call_llm(**call_kwargs)
        choice = response.choices[0]
        content = choice.message.content
        response_text = (content if isinstance(content, str) else str(content or "")).strip()
        from .compression import _coerce_int, _coerce_str
        finish_reason = _coerce_str(getattr(choice, "finish_reason", None))
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = _coerce_int(getattr(usage, "prompt_tokens", None))
            completion_tokens = _coerce_int(getattr(usage, "completion_tokens", None))
    except Exception as e:
        logger.warning("autobio CM call failed: %s", e)
        err = str(e)

    call_id: Optional[int] = None
    if engine._store is not None:
        try:
            call_id = engine._store.record_llm_call(
                purpose=purpose, prompt_messages=messages,
                response_text=response_text or None,
                model=cm_cfg.get("model") or engine.model,
                response_finish=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=err,
            )
        except Exception as rec_e:
            logger.error("autobio CM: failed to record llm_call: %s", rec_e)

    return (response_text, call_id)


# ---------------------------------------------------------------------------
# Op parsing and execution
# ---------------------------------------------------------------------------

_OP_HEADER_RE = re.compile(r"^\s*OPS:\s*$", re.MULTILINE)


def _split_reply(raw: str) -> Tuple[str, List[str]]:
    """Split CM's raw reply into natural-language text and op lines."""
    m = _OP_HEADER_RE.search(raw)
    if not m:
        return raw.strip(), []
    reply_text = raw[: m.start()].strip()
    ops_block = raw[m.end():].strip()
    op_lines = [ln.strip() for ln in ops_block.splitlines() if ln.strip()]
    return reply_text, op_lines


def _execute_ops(
    op_lines: List[str], engine: "AutobioContextEngine",
) -> List[Tuple[str, str]]:
    """Run each op, return a list of (op_line, result_text)."""
    results: List[Tuple[str, str]] = []
    for line in op_lines:
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        if not tokens:
            continue
        op = tokens[0].upper().rstrip(":")
        args = tokens[1:]
        try:
            res = _dispatch_op(op, args, engine)
        except Exception as e:
            res = f"error: {e}"
        results.append((line, res))
    return results


def _dispatch_op(op: str, args: List[str], engine: "AutobioContextEngine") -> str:
    store = engine._store
    if store is None:
        return "store not available"

    if op == "SEARCH":
        if not args:
            return "SEARCH requires a query"
        query = " ".join(args)
        # Route through search_all so FTS5 specials (quotes, parens,
        # dashes, *) get sanitized. Calling fts_messages/fts_summaries
        # directly with raw user input throws OperationalError on any
        # query containing those characters; the store helpers swallow
        # the error and return [], which would lie to CM that nothing
        # matched.
        from .search import search_all
        hits = search_all(store, query, max_results=10)
        if not hits:
            return f'no matches for "{query}"'
        lines = []
        for h in hits[:8]:
            snippet = (h.get("snippet") or "")[:120].replace("\n", " ")
            src = h.get("source", "?")
            lines.append(f"  - {src}: {snippet}...")
        return f'search "{query}" ({len(hits)} hits):\n' + "\n".join(lines)

    if op == "GREP":
        if not args:
            return "GREP requires a pattern"
        pattern = " ".join(args)
        from .search import grep_archive  # forward ref — added by task 9
        try:
            matches = grep_archive(store, pattern, limit=10)
        except Exception as e:
            return f"grep error: {e}"
        if not matches:
            return f'grep "{pattern}": no matches'
        return f'grep "{pattern}" ({len(matches)} hits):\n' + "\n".join(
            f"  - {m['source']}: {m['snippet']}" for m in matches
        )

    if op == "UNCOMPRESS":
        if not args:
            return "UNCOMPRESS requires a summary ID"
        sid = args[0]
        from .search import UnfoldFailed, expand_summary
        try:
            expanded = expand_summary(store, sid)
        except UnfoldFailed as fail:
            # Refuse to fabricate or silently truncate — surface the
            # incomplete-archive state so CM can communicate it to the
            # main instance and the user can decide how to proceed.
            return (
                f"UNCOMPRESS REFUSED for {sid}: {fail}. The archive is "
                f"missing some referenced source content; a partial unfold "
                f"would be misleading. The summary's text is still "
                f"available via memory_read."
            )
        except Exception as e:
            logger.exception("uncompress unexpected error: %s", e)
            return f"uncompress error: {e}"
        if not expanded:
            return f"no content expanded for {sid}"
        # Inline the expanded content in the op result — the main
        # instance sees it in the tool response, which becomes part of
        # its own context going forward. In-place live-list injection
        # is reserved for CM-on-behalf-of operations during compress().
        header = f"uncompressed {sid} → {len(expanded)} items:\n"
        body_lines = []
        for i, item in enumerate(expanded, 1):
            role = item.get("role", "?")
            content = item.get("content", "") or ""
            if isinstance(content, list):
                content = json.dumps(content)
            preview = content.strip()
            body_lines.append(f"  [{i}] ({role}) {preview}")
        return header + "\n".join(body_lines)

    if op == "PIN":
        if len(args) < 1:
            return "PIN requires a summary ID"
        sid = args[0]
        reason = " ".join(args[1:]) if len(args) > 1 else ""
        store.set_pin(sid, True)
        return f"pinned {sid}" + (f" ({reason})" if reason else "")

    if op == "UNPIN":
        if not args:
            return "UNPIN requires a summary ID"
        store.set_pin(args[0], False)
        return f"unpinned {args[0]}"

    if op == "SET_INTERVAL":
        if not args:
            return "SET_INTERVAL requires a number"
        try:
            n = int(args[0])
        except ValueError:
            return f"invalid interval: {args[0]}"
        engine.cfg.setdefault("cm", {})["checkin_interval"] = n
        store.kv_set("checkin_interval", n)
        return f"check-in interval set to {n} turns"

    if op == "MERGE":
        if not args:
            return "MERGE requires a level"
        try:
            level = int(args[0])
        except ValueError:
            return f"invalid level: {args[0]}"
        from .compression import ModelInfo, merge_if_ready
        mi = ModelInfo(
            model=engine.model, provider=engine.provider,
            base_url=engine.base_url, api_key=engine.api_key,
            api_mode=engine.api_mode,
        )
        # For v1 MERGE just runs the automatic pass which handles all levels.
        created = merge_if_ready(store, engine.cfg, mi)
        return f"merged: {created}" if created else "no merge — below threshold"

    if op == "MARK_DOCUMENT":
        if len(args) < 2:
            return "MARK_DOCUMENT requires at least <start_hash> <end_hash>"
        start_hash = args[0]
        end_hash = args[1]
        style = "chronological"
        title_tokens: List[str] = []
        for extra in args[2:]:
            if extra in ("chronological", "monolithic"):
                style = extra
            else:
                title_tokens.append(extra)
        title = " ".join(title_tokens) if title_tokens else None
        kind = f"doc_{style}"
        # Validate hashes exist in the archive.
        rows = store.get_messages_by_hashes([start_hash, end_hash])
        found = {r["hash"] for r in rows}
        missing = [h for h in (start_hash, end_hash) if h not in found]
        if missing:
            return f"cannot MARK_DOCUMENT: unknown hashes {missing}"
        meta = {
            "title": title,
            "source": "cm",
        }
        pin_id = store.add_pin(
            start_hash, end_hash,
            reason=title or "document",
            kind=kind, meta=meta,
        )
        return f"marked document pin {pin_id} ({kind}){': ' + title if title else ''}"

    if op == "LIST_DOCUMENTS":
        docs = [p for p in store.list_pins() if str(p.get("kind", "")).startswith("doc_")]
        if not docs:
            return "no documents currently marked"
        lines = [f"{len(docs)} documents:"]
        for d in docs:
            meta = d.get("meta") or {}
            title = meta.get("title") if isinstance(meta, dict) else None
            origin_tokens = meta.get("original_tokens") if isinstance(meta, dict) else None
            lines.append(
                f"  - pin {d['id']}: kind={d['kind']}"
                + (f" title={title!r}" if title else "")
                + (f" original~{origin_tokens}t" if origin_tokens else "")
                + f" start={d['start_hash']} end={d['end_hash']}"
            )
        return "\n".join(lines)

    if op == "UNMARK_DOCUMENT":
        if not args:
            return "UNMARK_DOCUMENT requires a pin_id"
        try:
            pin_id = int(args[0])
        except ValueError:
            return f"invalid pin_id: {args[0]}"
        pins = [p for p in store.list_pins() if p["id"] == pin_id]
        if not pins:
            return f"pin {pin_id} not found"
        if not str(pins[0].get("kind", "")).startswith("doc_"):
            return f"pin {pin_id} is not a document"
        store.remove_pin(pin_id)
        return f"unmarked document pin {pin_id} (L1 summaries retained in archive)"

    if op == "DEFER":
        if not args:
            return "DEFER requires a turn count"
        try:
            n = int(args[0])
        except ValueError:
            return f"invalid turn count: {args[0]}"
        engine._defer_checkin_for(n)
        reason = " ".join(args[1:])
        return f"deferred check-in by {n} turns" + (f" ({reason})" if reason else "")

    if op == "RESUME_COMPRESSION":
        # Clears a halt flag set when the summarizer returned empty. Use
        # this ONLY after the underlying cause has been addressed (model
        # swap, quota top-up, etc.). Calling it with the summarizer
        # still broken just reintroduces the halt on the next threshold.
        if not getattr(engine, "_compression_halted", False):
            return "compression is not currently halted — no-op"
        prior = getattr(engine, "_halt_reason", None)
        engine.resume_compression()
        return f"compression halt cleared (prior reason: {prior})"

    return f"unknown op: {op}"


def _format_op_results(results: List[Tuple[str, str]]) -> str:
    parts = ["(operations executed)"]
    for line, res in results:
        parts.append(f"▸ {line}")
        parts.append(textwrap.indent(res, "  "))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# YAML-lite emitter
# ---------------------------------------------------------------------------

def _dump_yaml_lite(obj: Any, indent: int = 0) -> str:
    """Minimal YAML serializer for dashboard dicts.

    Avoids a PyYAML dependency for this single use. Handles dict, list,
    and scalars — no anchors, no multi-line strings.
    """
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(_dump_yaml_lite(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                sub = _dump_yaml_lite(item, indent + 1).lstrip()
                lines.append(f"{pad}- {sub}")
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_scalar(obj)}"


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # Quote strings only if needed.
        if re.match(r"^[\w\-./%:]+$", v):
            return v
        return json.dumps(v)
    return json.dumps(v)

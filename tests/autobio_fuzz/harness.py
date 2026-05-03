"""Harness runner.

Takes a list of ``Turn`` steps, mutates a live message list, drives a
real ``AutobioContextEngine`` (with a real LLM for summarization),
runs validators after every step, and snapshots the state on every
transition point (fold, merge, speculative commit, pin retire) into a
markdown report.

Failures are collected, not raised. Returns a ``RunResult`` so the
caller can aggregate across many scenarios.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plugins.autobio.engine import AutobioContextEngine
from plugins.autobio.ids import message_hash

from .steps import (
    AddAssistant, AddToolCall, AddUser, CallCM,
    MarkPin, Rollback, Step, TriggerCompress, TriggerProactiveCheckin,
    Turn, UnmarkPin,
)
from .validators import Issue, run_all, validate_archive, validate_message_list


@dataclass
class ArchiveSnapshot:
    messages_count: int
    summary_counts: Dict[int, int]
    pin_count: int
    pin_kinds: Dict[str, int]

    @classmethod
    def of(cls, store) -> "ArchiveSnapshot":
        conn = getattr(store, "_conn", None) or getattr(store, "conn", None)
        msg_count = 0
        sc: Dict[int, int] = {}
        pin_kinds: Dict[str, int] = {}
        pin_count = 0
        if conn is not None:
            try:
                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages"
                ).fetchone()[0]
            except Exception:
                pass
            try:
                rows = conn.execute(
                    "SELECT level, COUNT(*) FROM summaries "
                    "WHERE merged_into IS NULL GROUP BY level"
                ).fetchall()
                sc = {int(lvl): int(cnt) for lvl, cnt in rows}
            except Exception:
                pass
            try:
                rows = conn.execute(
                    "SELECT kind, COUNT(*) FROM pins GROUP BY kind"
                ).fetchall()
                for k, c in rows:
                    pin_kinds[k or "pin"] = int(c)
                    pin_count += int(c)
            except Exception:
                pass
        return cls(msg_count, sc, pin_count, pin_kinds)

    def diff(self, other: "ArchiveSnapshot") -> Dict[str, Any]:
        delta: Dict[str, Any] = {}
        if self.messages_count != other.messages_count:
            delta["messages"] = other.messages_count - self.messages_count
        level_delta: Dict[int, int] = {}
        all_levels = set(self.summary_counts) | set(other.summary_counts)
        for lvl in sorted(all_levels):
            before = self.summary_counts.get(lvl, 0)
            after = other.summary_counts.get(lvl, 0)
            if before != after:
                level_delta[lvl] = after - before
        if level_delta:
            delta["summaries"] = level_delta
        if self.pin_count != other.pin_count:
            delta["pins"] = other.pin_count - self.pin_count
        if self.pin_kinds != other.pin_kinds:
            delta["pin_kinds"] = {
                "before": self.pin_kinds, "after": other.pin_kinds,
            }
        return delta


@dataclass
class TransitionEvent:
    step_index: int
    turn_index: int
    kind: str              # "compress" | "speculative" | "pin_retire" | "merge"
    snapshot_before: ArchiveSnapshot
    snapshot_after: ArchiveSnapshot
    pre_tail: List[Dict[str, Any]]
    post_tail: List[Dict[str, Any]]
    new_summaries: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass
class RunResult:
    scenario_name: str
    steps_run: int
    transitions: List[TransitionEvent] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    crashed: Optional[str] = None
    duration_s: float = 0.0
    report_path: Optional[str] = None
    cm_transcript: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    tail_tokens: int = 15_000
    # Weighted chunk budget. With code/console at 0.1× and dialogue at
    # 1.0×, 5_000 weighted ≈ 50K raw tokens of file reads or 5K of
    # dialogue. The picker accumulates weighted tokens; the LLM still
    # sees the actual content. See plugins/autobio/config.py weights.
    chunk_tokens: int = 5_000
    summary_target_tokens: int = 800
    # Reasoning models burn a large hidden completion budget before
    # emitting content. K2.6 needs more headroom on chunks that
    # contain large tool results (file reads, log dumps) — a single
    # 22 KB tool result + reasoning easily exhausts 4000. 8000 covers
    # most realistic chunks; sustained empty across retries indicates
    # the auxiliary genuinely can't summarize this content (e.g.,
    # content-filter, refusal pattern), which the halt mechanism
    # surfaces correctly.
    max_tokens_override: int = 8_000
    merge_threshold: int = 3
    model: str = "moonshotai/kimi-k2.6"
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    hermes_home: Optional[str] = None
    report_dir: Optional[str] = None
    auto_doc_threshold_tokens: int = 12_000   # lower than default so scenarios hit it
    speculative: bool = True
    # Redact API keys / secrets from incoming step content before they
    # hit the engine. The fuzz harness routes through OpenRouter, so a
    # transcript replay containing keys would leak them to the
    # destination provider. Patterns are conservative (Anthropic /
    # OpenRouter / OpenAI / GitHub / AWS / JWT / env-style assignments).
    # Disable when fuzzing in a same-provider configuration where keys
    # are expected to round-trip.
    redact_secrets: bool = True


def _new_engine(cfg: HarnessConfig) -> AutobioContextEngine:
    eng = AutobioContextEngine()
    eng.update_model(
        model=cfg.model,
        context_length=int(cfg.tail_tokens / 0.75),  # derive context from desired tail
        provider=cfg.provider,
        api_key=os.environ.get(cfg.api_key_env, ""),
    )
    home = cfg.hermes_home or tempfile.mkdtemp(prefix="autobio_fuzz_")
    session_id = f"fuzz_{int(time.time() * 1000) % 1_000_000}"
    eng.on_session_start(session_id, hermes_home=home)

    # Install CM transcript capture: every cm_respond / proactive_checkin
    # call appends an event dict to this list. The report writer renders
    # the list as a "CM dialogues" section so the user can read what CM
    # actually saw and produced — the highest-signal way to evaluate
    # whether CM behaves like a useful collaborator.
    eng._cm_transcript = []

    eng.cfg["compression"]["chunk_tokens"] = cfg.chunk_tokens
    eng.cfg["compression"]["summary_target_tokens"] = cfg.summary_target_tokens
    eng.cfg["compression"]["max_tokens_override"] = cfg.max_tokens_override
    eng.cfg["compression"]["merge_threshold"] = cfg.merge_threshold
    eng.cfg["compression"]["speculative"] = cfg.speculative
    if not cfg.speculative and eng._speculative is not None:
        eng._speculative._enabled = False
    eng.cfg["cm"]["model"] = cfg.model
    eng.cfg["documents"]["auto_threshold_tokens"] = cfg.auto_doc_threshold_tokens
    # Protect only the system message; we want the rest to flow through.
    eng.protect_first_n = 1
    # Override MINIMUM_CONTEXT_LENGTH clamp so the fuzz runs at the
    # small sizes the user wants. In prod update_model() applies a
    # 64k floor which is fine for real models but defeats the fuzz
    # harness, which wants compression to fire quickly and often.
    eng.threshold_tokens = cfg.tail_tokens
    return eng


def run_scenario(
    name: str,
    turns: Sequence[Step],
    *,
    config: HarnessConfig = HarnessConfig(),
    report_path: Optional[str] = None,
) -> RunResult:
    """Execute a scenario end-to-end against a real engine.

    Returns a RunResult with all validator findings and a transition
    log. A report markdown file is written to ``report_path`` (or the
    configured report_dir).
    """
    start = time.time()
    result = RunResult(scenario_name=name, steps_run=0)
    eng = _new_engine(config)
    store = eng._store
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content":
         "You are a backend engineer collaborating with a colleague."},
    ]
    turn_counter = 0

    # Make sure the kv starts at turn 0.
    store.kv_set("current_turn", 0)
    store.kv_set("first_live_turn", 0)

    try:
        for step_index, step in enumerate(turns):
            if not isinstance(step, Turn):
                step = Turn([step])
            turn_counter += 1
            store.kv_set("current_turn", turn_counter)

            snap_before = ArchiveSnapshot.of(store)
            pre_live = list(messages)

            # Apply sub-steps.
            for sub in step.substeps:
                try:
                    _apply(sub, messages, eng, redact=config.redact_secrets)
                except Exception as e:
                    result.issues.append(Issue(
                        "error", "apply_exception",
                        f"step {step_index} substep {type(sub).__name__}: {e}",
                        {"traceback": traceback.format_exc()},
                    ))

            # Fire on_turn_end as the real runner does. This may mutate
            # messages (proactive CM inject) and may trigger speculation.
            try:
                eng.on_turn_end(messages)
            except Exception as e:
                result.issues.append(Issue(
                    "error", "on_turn_end",
                    f"step {step_index} on_turn_end raised: {e}",
                    {"traceback": traceback.format_exc()},
                ))

            # Mimic the real runner: if preflight says we're at/over
            # threshold, compress. Without this, the archive never gets
            # populated (archive writes happen inside compress()).
            #
            # Cap iterations: if compress() fails to shrink the list
            # (e.g., summarizer returns empty repeatedly), one turn
            # could loop forever. Two attempts is plenty — the surface
            # is the bug, not the loop.
            try:
                attempts = 0
                while eng.should_compress_preflight(messages) and attempts < 2:
                    before_len = len(messages)
                    new_messages = eng.compress(messages)
                    messages[:] = new_messages
                    attempts += 1
                    if len(messages) >= before_len:
                        result.issues.append(Issue(
                            "warn", "compress_no_progress",
                            f"step {step_index} compress did not shrink list",
                            {"before": before_len, "after": len(messages),
                             "attempts": attempts},
                        ))
                        break
            except Exception as e:
                result.issues.append(Issue(
                    "error", "auto_compress",
                    f"step {step_index} auto-compress raised: {e}",
                    {"traceback": traceback.format_exc()},
                ))

            snap_after = ArchiveSnapshot.of(store)
            delta = snap_before.diff(snap_after)

            # Validate the live list every step.
            for iss in validate_message_list(messages):
                iss.context.setdefault("step_index", step_index)
                result.issues.append(iss)

            if delta:
                ev = TransitionEvent(
                    step_index=step_index,
                    turn_index=turn_counter,
                    kind=_classify_transition(delta),
                    snapshot_before=snap_before,
                    snapshot_after=snap_after,
                    pre_tail=_compact(pre_live[-8:]),
                    post_tail=_compact(messages[-8:]),
                    new_summaries=_latest_summaries(store, snap_before, snap_after),
                    notes=json.dumps(delta, default=str),
                )
                result.transitions.append(ev)
                # Full archive check at transitions.
                for iss in validate_archive(store):
                    iss.context.setdefault("step_index", step_index)
                    iss.context.setdefault("transition", ev.kind)
                    result.issues.append(iss)

            result.steps_run = step_index + 1

            # If compression halted, stop the replay. There's no point
            # marching the remaining pre-recorded turns — every one
            # would just accumulate raw messages without folding, and
            # we already captured the halt event. The user wants the
            # report to reflect "halted at turn N", not "halted then
            # mechanically iterated 600 more empty turns".
            if getattr(eng, "_compression_halted", False):
                logger.warning(
                    "fuzz harness: compression halted at step %d — "
                    "exiting replay loop early. Halt reason: %s",
                    step_index, getattr(eng, "_halt_reason", "?"),
                )
                result.issues.append(Issue(
                    "warn", "halted_replay_early_exit",
                    f"replay aborted at step {step_index} due to compression halt",
                    {"halt_reason": getattr(eng, "_halt_reason", None),
                     "remaining_turns": len(turns) - (step_index + 1)},
                ))
                break

    except Exception as e:
        result.crashed = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Final sweep.
        for iss in validate_archive(store):
            iss.context.setdefault("final_pass", True)
            result.issues.append(iss)
        # Copy CM transcript out before tearing down the engine.
        try:
            result.cm_transcript = list(getattr(eng, "_cm_transcript", []) or [])
        except Exception:
            result.cm_transcript = []
        try:
            eng.on_session_end(eng._session_id or "fuzz", messages)
        except Exception:
            pass
        result.duration_s = time.time() - start

    report_path = report_path or _default_report_path(name, config)
    _write_report(result, config, report_path)
    result.report_path = report_path
    return result


# ---------------------------------------------------------------------------
# Step application
# ---------------------------------------------------------------------------

def _apply(step: Step, messages: List[Dict[str, Any]],
           eng: AutobioContextEngine,
           redact: bool = True) -> None:
    """Apply one step to the live message list.

    When ``redact`` is True, all string content (user message,
    assistant message, tool args, tool result) flowing through the
    step is run through the secrets redactor before it hits the
    engine. This is critical for cross-provider replay (e.g.
    Anthropic transcript fuzzed via OpenRouter) — without it, API
    keys mentioned anywhere in the source conversation get forwarded
    to the destination provider.
    """
    from plugins.autobio.ingest import redact_text as _red
    def _r(s: Any) -> Any:
        if not redact or not isinstance(s, str):
            return s
        return _red(s)

    if isinstance(step, AddUser):
        messages.append({"role": "user", "content": _r(step.content)})
    elif isinstance(step, AddAssistant):
        messages.append({"role": "assistant", "content": _r(step.content)})
    elif isinstance(step, AddToolCall):
        # Redact both arguments (could be a payload echoing a secret)
        # and the result (file contents, env-var dumps, etc.).
        args_str = _r(json.dumps(step.arguments)) if redact else json.dumps(step.arguments)
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": step.call_id,
                "type": "function",
                "function": {
                    "name": step.tool_name,
                    "arguments": args_str,
                },
            }],
        }
        messages.append(assistant)
        if step.result is not None:
            messages.append({
                "role": "tool",
                "tool_call_id": step.call_id,
                "name": step.tool_name,
                "content": _r(step.result),
            })
    elif isinstance(step, Rollback):
        # Drop the last n messages, but never touch the system.
        n = min(step.n, max(0, len(messages) - 1))
        if n:
            del messages[-n:]
    elif isinstance(step, MarkPin):
        if eng._store is None:
            return
        si = max(0, min(step.start_index, len(messages) - 1))
        ei = max(0, min(step.end_index, len(messages) - 1))
        if si > ei:
            si, ei = ei, si
        start_msg = messages[si]
        end_msg = messages[ei]
        start_h = message_hash(start_msg)
        end_h = message_hash(end_msg)
        # Archive if not already present — add_pin requires hashes to
        # resolve in the messages table for the validator to pass.
        eng._store.upsert_message(start_h, start_msg, turn=si, position=si)
        eng._store.upsert_message(end_h, end_msg, turn=ei, position=ei)
        eng._store.add_pin(
            start_h, end_h, reason=step.reason,
            kind=step.kind, meta=step.meta or {"source": "fuzz"},
        )
    elif isinstance(step, UnmarkPin):
        if eng._store is None:
            return
        target_kind = step.kind
        for p in eng._store.list_pins():
            if str(p.get("kind", "")) == target_kind:
                # Tolerant match: any pin of this kind covering the
                # requested index range. For retired docs this may
                # find nothing — harness treats that as a no-op.
                eng._store.remove_pin(p["id"])
                return
    elif isinstance(step, TriggerCompress):
        new_messages = eng.compress(messages)
        messages[:] = new_messages
    elif isinstance(step, CallCM):
        # Route through the same path the main instance uses.
        reply = eng.handle_tool_call(
            "cm_manage_context",
            {"request": step.request},
            messages=messages,
        )
        # Optionally append CM's reply to the live list as a tool result
        # so the main instance "sees" it. We don't have a real tool_call_id
        # since this is a forced invocation; skip the append — the CM
        # transcript captures everything we want.
        del reply
    elif isinstance(step, TriggerProactiveCheckin):
        # Clear interval/defer barriers so the check-in actually fires.
        if eng._store is not None:
            eng._store.kv_set("last_checkin_turn", -10_000)
            eng._store.kv_set("deferred_until_turn", 0)
        injected = eng.maybe_proactive_checkin(messages)
        if injected is not None:
            messages.append(injected)
    else:
        raise TypeError(f"unknown step type: {type(step).__name__}")


# ---------------------------------------------------------------------------
# Transition classification + snapshot helpers
# ---------------------------------------------------------------------------

def _classify_transition(delta: Dict[str, Any]) -> str:
    levels = delta.get("summaries") or {}
    if 3 in levels and levels[3] > 0:
        return "merge_l2_to_l3"
    if 2 in levels and levels[2] > 0:
        return "merge_l1_to_l2"
    if 1 in levels and levels[1] > 0:
        # Could be either compress-driven or speculative; we can't
        # distinguish from delta alone (both look the same).
        return "l1_commit"
    if "pins" in delta and delta["pins"] < 0:
        return "pin_retire"
    if "pins" in delta and delta["pins"] > 0:
        return "pin_add"
    if "messages" in delta:
        return "archive_grow"
    return "other"


def _latest_summaries(store, before: ArchiveSnapshot,
                      after: ArchiveSnapshot) -> List[Dict[str, Any]]:
    """Return summary rows created between ``before`` and ``after``."""
    out: List[Dict[str, Any]] = []
    conn = getattr(store, "_conn", None) or getattr(store, "conn", None)
    if conn is None:
        return out
    added_levels = set()
    for lvl in sorted(set(before.summary_counts) | set(after.summary_counts)):
        if after.summary_counts.get(lvl, 0) > before.summary_counts.get(lvl, 0):
            added_levels.add(lvl)
    if not added_levels:
        return out
    placeholders = ",".join("?" * len(added_levels))
    try:
        rows = conn.execute(
            f"SELECT id, level, content, tokens, tags "
            f"FROM summaries WHERE level IN ({placeholders}) "
            f"ORDER BY created_at DESC LIMIT 6",
            tuple(added_levels),
        ).fetchall()
    except Exception:
        return out
    cols = ("id", "level", "content", "tokens", "tags")
    for row in rows:
        rec = dict(zip(cols, row))
        # Derive turn range on demand — never stored on the summary.
        try:
            first, last = store.get_summary_turn_range(rec["id"])
            rec["first_turn"] = first
            rec["last_turn"] = last
        except Exception:
            rec["first_turn"] = None
            rec["last_turn"] = None
        out.append(rec)
    return out


def _compact(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip long content fields so the report stays readable."""
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            out.append({"role": "??", "content": repr(m)[:80]})
            continue
        copy = {"role": m.get("role", "?")}
        content = m.get("content")
        if isinstance(content, str):
            copy["content"] = content if len(content) < 140 else content[:120] + f"…[{len(content)}c]"
        else:
            copy["content"] = repr(content)[:120]
        if m.get("tool_calls"):
            copy["tool_calls"] = [
                {"id": (tc.get("id") if isinstance(tc, dict) else None),
                 "name": (tc.get("function", {}).get("name") if isinstance(tc, dict) else None)}
                for tc in m.get("tool_calls") or []
            ]
        if m.get("tool_call_id"):
            copy["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            copy["name"] = m["name"]
        out.append(copy)
    return out


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _default_report_path(name: str, cfg: HarnessConfig) -> str:
    base = Path(cfg.report_dir or
                os.path.join(os.path.dirname(__file__), "reports"))
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(base / f"{stamp}_{name}.md")


def _write_report(result: RunResult, cfg: HarnessConfig, path: str) -> None:
    lines: List[str] = []
    push = lines.append
    push(f"# Fuzz scenario: `{result.scenario_name}`")
    push("")
    push(f"- Model: `{cfg.model}` via `{cfg.provider}`")
    push(f"- tail/chunk/merge: {cfg.tail_tokens}/{cfg.chunk_tokens}/{cfg.merge_threshold}")
    push(f"- Steps run: {result.steps_run}")
    push(f"- Duration: {result.duration_s:.2f}s")
    push(f"- Issues: {result.errors} errors, {result.warnings} warnings")
    push(f"- Transitions: {len(result.transitions)}")
    if result.crashed:
        push("")
        push("## CRASHED")
        push("```")
        push(result.crashed)
        push("```")
    push("")

    push("## Issues")
    if not result.issues:
        push("_none_")
    else:
        for iss in result.issues:
            push(f"- {iss}")
    push("")

    push("## CM dialogues")
    if not result.cm_transcript:
        push("_none — CM was never invoked or only via the halt-message path._")
    else:
        push(f"_{len(result.cm_transcript)} CM events; reproduced verbatim below._")
        push("")
        for i, ev in enumerate(result.cm_transcript, 1):
            kind = ev.get("kind", "?")
            decision = ev.get("decision")
            header = f"### CM event {i} — {kind}"
            if decision:
                header += f" → {decision}"
            push(header)
            push("")
            req = ev.get("request") or "(none)"
            push("**Request:**")
            push("```")
            push(req[:400] + ("…" if len(req) > 400 else ""))
            push("```")
            dash = ev.get("dashboard") or {}
            if dash:
                push("**Dashboard digest:**")
                push("```yaml")
                # Pull the most readable bits, keep it bounded.
                tail = dash.get("tail", {})
                ctx = dash.get("context", {})
                layers = dash.get("layers", {})
                comp = dash.get("compression", {})
                push(f"tail: tokens={tail.get('tokens')} msgs={tail.get('message_count')}")
                push(f"context: usage={ctx.get('usage_percent')}% limit={ctx.get('limit')} threshold={ctx.get('threshold')}")
                if comp.get("halted"):
                    push(f"compression: HALTED — {comp.get('halt_reason')}")
                for lvl in ("l1", "l2", "l3"):
                    info = layers.get(lvl) or {}
                    if info.get("count"):
                        previews = info.get("recent") or []
                        push(f"{lvl}: count={info['count']} tokens={info.get('total_tokens')} newest={info.get('newest')}")
                        for p in previews[-3:]:
                            preview_txt = (p.get("preview") or "")[:140]
                            push(f"  - {p.get('id')}: {preview_txt}")
                push("```")
            raw = ev.get("raw_reply") or ""
            reply = ev.get("reply_text") or ""
            push("**CM raw reply:**")
            push("```")
            push(raw[:1500] + (f"\n…[{len(raw)} chars total]" if len(raw) > 1500 else ""))
            push("```")
            ops = ev.get("ops") or []
            if ops:
                push("**Ops executed:**")
                for line, res in ops:
                    push(f"- `{line}`")
                    push(f"  → {(res or '')[:300]}")
            push("")

    push("## Transitions")
    if not result.transitions:
        push("_none_")
    else:
        for i, ev in enumerate(result.transitions):
            push(f"### {i+1}. step {ev.step_index} (turn {ev.turn_index}) — {ev.kind}")
            push("")
            push(f"- delta: `{ev.notes}`")
            push(f"- before: msgs={ev.snapshot_before.messages_count} "
                 f"summaries={ev.snapshot_before.summary_counts} "
                 f"pins={ev.snapshot_before.pin_count}")
            push(f"- after:  msgs={ev.snapshot_after.messages_count} "
                 f"summaries={ev.snapshot_after.summary_counts} "
                 f"pins={ev.snapshot_after.pin_count}")
            if ev.new_summaries:
                push("")
                push("**New summaries:**")
                for s in ev.new_summaries:
                    tags = s.get("tags") or "[]"
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        pass
                    push(f"- `{s['id']}` (L{s['level']}, {s['tokens']} tok, "
                         f"turns {s.get('first_turn')}..{s.get('last_turn')}, "
                         f"tags={tags})")
                    content = s.get("content") or ""
                    preview = content[:600].replace("\n", " ")
                    if len(content) > 600:
                        preview += f" …[{len(content)}c total]"
                    push(f"  > {preview}")
            push("")
            push("**Pre-tail (last 8):**")
            push("```json")
            push(json.dumps(ev.pre_tail, indent=2, ensure_ascii=False))
            push("```")
            push("**Post-tail (last 8):**")
            push("```json")
            push(json.dumps(ev.post_tail, indent=2, ensure_ascii=False))
            push("```")
            push("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_many(
    scenarios: Sequence[Any],
    *,
    config: HarnessConfig = HarnessConfig(),
) -> List[RunResult]:
    """Run a list of ``(name, turns)`` tuples and return all results."""
    results: List[RunResult] = []
    for name, turns in scenarios:
        print(f"[fuzz] running {name} ({len(turns)} turns)...", flush=True)
        r = run_scenario(name, turns, config=config)
        status = "OK" if r.errors == 0 and not r.crashed else "FAIL"
        print(f"  {status}  errors={r.errors} warns={r.warnings} "
              f"transitions={len(r.transitions)} report={r.report_path}",
              flush=True)
        results.append(r)
    return results

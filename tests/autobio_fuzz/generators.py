"""Scenario generators for the autobio fuzz harness.

Two modes:

- :func:`scripted_narrative` — deterministic, coherent conversation
  following a named script (``debug_thread``, ``research_thread``,
  ``log_reading_thread``). Edge cases are injected at fixed offsets so
  specific hazards always get exercised. Output is eyeballable: you
  can read the summary of turns 1..N and tell if it makes sense.

- :func:`random_fuzz` — weighted action stream seeded by ``seed``.
  Content is pulled from a varied corpus (code, logs, JSON, unicode,
  empty). Summaries from these runs are intentionally meaningless —
  this mode is for stressing structural invariants, not quality.

Both modes emit lists of ``Turn`` steps so the harness can fire
``on_turn_end`` at natural boundaries.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from .steps import (
    AddAssistant, AddToolCall, AddUser,
    MarkPin, Rollback, Step, TriggerCompress, Turn, UnmarkPin,
)


# ---------------------------------------------------------------------------
# Scripted narratives
# ---------------------------------------------------------------------------

_DEBUG_OPENING = [
    "We've got a weird 503 rate on /api/rate in prod — started about an hour ago.",
    "Checked the pods — all 12 are up. CPU normal, memory normal.",
    "Can you look at the Redis connection pool config?",
    "Pool max is set to 10 in config.yaml. That feels low for this pod count.",
    "Right — if each pod opens 2-3 connections, we're saturating the pool immediately.",
    "Rollout log says deploy finished at 12:00, first 503s at 12:03.",
]

_DEBUG_BODY = [
    ("Let me check auth_validator.check_token().", "assistant"),
    ("grep -n 'issuer' auth_validator.py", "tool"),
    ("What's the `issuer` claim actually being compared to?", "user"),
    ("Looks like we strip trailing slashes on one side only.",  "assistant"),
    ("cat /var/log/app/rate-limiter.log | tail -200", "tool"),
    ("So the 401 bug on /api/rate is a separate issue from the 503s.", "user"),
    ("Right — 503 is connection pool exhaustion, 401 is token normalization.", "assistant"),
    ("Can you sketch the fix for both?", "user"),
    ("For 503: bump pool max to 50 in config.yaml, rolling restart.", "assistant"),
    ("For 401: normalize issuer in check_token() before comparing.", "assistant"),
    ("Test locally first — tests/auth/test_check_token.py has the regression suite.", "user"),
    ("Running it now.", "assistant"),
    ("pytest tests/auth/test_check_token.py -v", "tool"),
    ("All green. Rolling out 2.3.2 with both fixes.", "assistant"),
    ("kubectl rollout restart deployment/rate-limiter", "tool"),
    ("Incident declared resolved at 12:17, postmortem scheduled for tomorrow.", "assistant"),
]


def _fab_log(n_lines: int, marker: str = "LOG") -> str:
    lines: List[str] = []
    for i in range(n_lines):
        lines.append(
            f"{marker} [2026-04-{10 + (i % 20):02d} "
            f"{(i * 7) % 24:02d}:{(i * 13) % 60:02d}:{(i * 29) % 60:02d}] "
            f"request_id=req-{i:06d} user=user-{i % 37} "
            f"duration_ms={100 + (i * 17) % 2000} status="
            f"{200 if i % 7 else 503}"
        )
    return "\n".join(lines)


def scripted_narrative(
    variant: str = "debug_thread",
    *,
    seed: int = 0,
    target_turns: int = 40,
) -> List[Step]:
    """Produce a deterministic Turn list for a named scenario variant.

    Variants:

    - ``debug_thread``: the 503 + 401 debugging arc above, plus
      follow-up chatter to hit target_turns. Edges injected: a
      rollback after the first fix proposal, a pin on the fix-sketch
      turns, a compress forced at the midpoint.
    - ``research_thread``: a multi-topic research conversation that
      drifts between unrelated areas; tests summary coherence across
      topic boundaries.
    - ``log_reading_thread``: a single giant read_file result is
      ingested (auto-marked as doc_chronological), then the agent
      reasons about it over many turns; tests doc-fold semantics
      with a real chunk pipeline.
    """
    rng = random.Random(seed)
    turns: List[Turn] = []

    if variant == "debug_thread":
        turns.append(Turn([AddUser(_DEBUG_OPENING[0])]))
        turns.append(Turn([AddAssistant(_DEBUG_OPENING[1])]))
        turns.append(Turn([AddUser(_DEBUG_OPENING[2])]))
        turns.append(Turn([AddAssistant(_DEBUG_OPENING[3])]))
        turns.append(Turn([AddUser(_DEBUG_OPENING[4])]))
        turns.append(Turn([AddAssistant(_DEBUG_OPENING[5])]))

        cid = 0
        for content, role in _DEBUG_BODY:
            cid += 1
            if role == "user":
                turns.append(Turn([AddUser(content)]))
            elif role == "assistant":
                turns.append(Turn([AddAssistant(content)]))
            elif role == "tool":
                # assistant thinks -> calls tool -> tool result -> assistant summarizes
                call_id = f"c{cid:04d}"
                result = _fab_log(30, marker="APP")
                turns.append(Turn([
                    AddToolCall(
                        tool_name="terminal",
                        arguments={"command": content},
                        result=result,
                        call_id=call_id,
                    ),
                ]))

        # Inject hazards at fixed offsets ----------------------------
        # (a) pin the fix-sketch turns (offset ~18 in the turn list).
        pin_offset = min(18, max(5, len(turns) - 8))
        turns.insert(pin_offset, Turn([MarkPin(
            start_index=_resolve_relative(pin_offset - 3),
            end_index=_resolve_relative(pin_offset),
            kind="pin", reason="keep fix sketch",
        )]))

        # (b) one rollback mid-arc.
        rollback_offset = min(26, max(8, len(turns) // 2 + 4))
        turns.insert(rollback_offset, Turn([Rollback(n=2)]))

        # (c) add some filler turns to reach target_turns. Mix of
        # single assistant thoughts and small tool calls to grow tail
        # past the compression threshold.
        while len(turns) < target_turns:
            k = len(turns)
            if k % 5 == 0:
                cid += 1
                turns.append(Turn([
                    AddToolCall(
                        tool_name="read_file",
                        arguments={"path": f"src/module_{k}.py"},
                        result=_fab_log(80 + rng.randint(0, 40), marker="SRC"),
                        call_id=f"f{cid:04d}",
                    ),
                ]))
            else:
                # Pad user/assistant turns so token count grows at a
                # useful rate for small fuzz budgets. Content stays
                # readable; the repeated reasoning sentence is the
                # only filler.
                filler = " Context: " + " ".join(
                    [_reasoning(rng) for _ in range(8)]
                )
                turns.append(Turn([
                    AddUser(f"Follow-up {k}: what about {_topic(rng)}?" + filler),
                ]))
                turns.append(Turn([
                    AddAssistant(
                        f"For {_topic(rng)}: we should {_action(rng)}. "
                        f"That ties into {_topic(rng)} because "
                        f"{_reasoning(rng)}." + filler
                    ),
                ]))
        return turns

    if variant == "log_reading_thread":
        # One huge tool result followed by many turns of reasoning.
        turns.append(Turn([
            AddUser("Read /var/log/prod/incident-2026-04-10.log and summarize."),
        ]))
        # ~250 log lines → well over the auto-threshold so the tool
        # result becomes a doc_chronological pin automatically.
        huge_log = _fab_log(600, marker="PROD")
        turns.append(Turn([
            AddToolCall(
                tool_name="read_file",
                arguments={"path": "/var/log/prod/incident-2026-04-10.log"},
                result=huge_log,
                call_id="read-log-1",
            ),
        ]))
        for i in range(target_turns):
            if i % 3 == 0:
                turns.append(Turn([AddUser(
                    f"What does the log say about {_log_subject(rng)} around turn {i}?"
                )]))
            else:
                turns.append(Turn([AddAssistant(
                    f"Looking at the log: {_log_subject(rng)} appears around "
                    f"line {100 + i * 17}. " + "Additional context. " * 15
                )]))
        return turns

    if variant == "research_thread":
        # Topic drift across four areas — cache, TLS, Kafka, auth.
        topics = ["cache eviction", "TLS handshake latency",
                  "Kafka consumer lag", "JWT audience validation"]
        i = 0
        cid = 0
        while len(turns) < target_turns:
            topic = topics[i % len(topics)]
            turns.append(Turn([AddUser(
                f"Moving on — what do we know about {topic}?"
            )]))
            turns.append(Turn([AddAssistant(
                f"For {topic}: " + _reasoning(rng) + ". " +
                "Worth checking " + _action(rng) + "."
            )]))
            cid += 1
            turns.append(Turn([AddToolCall(
                tool_name="memory_search",
                arguments={"query": topic},
                result="hits: " + ", ".join(
                    f"L1-{j:04d}" for j in range(3)
                ) + "\n" + _fab_log(20, marker="RES"),
                call_id=f"s{cid:04d}",
            )]))
            i += 1
        return turns

    raise ValueError(f"unknown variant: {variant!r}")


def _resolve_relative(idx: int) -> int:
    # Pin indices inside a scripted sequence — absolute position in
    # the live list is resolved at harness-execution time, but for
    # now we emit a placeholder that counts from the system prompt
    # (1) onward. Generator defines the intent; harness maps to real
    # message indices after building the list.
    return max(1, idx + 1)


_TOPICS = [
    "token refresh", "pool sizing", "idempotency keys",
    "circuit breakers", "dead-letter queues", "clock skew",
    "backpressure", "eventual consistency", "TLS pinning",
]
_ACTIONS = [
    "add a metric", "increase the timeout", "audit the retry policy",
    "pin a specific version", "add a feature flag", "write a regression test",
]
_REASONING = [
    "the current behavior leaks under high load",
    "the retry policy compounds the backpressure",
    "downstream assumes we always send a correlation id",
    "our clocks can drift up to 30s from NTP",
    "the library changed its default in v2",
]


def _topic(rng: random.Random) -> str:
    return rng.choice(_TOPICS)


def _action(rng: random.Random) -> str:
    return rng.choice(_ACTIONS)


def _reasoning(rng: random.Random) -> str:
    return rng.choice(_REASONING)


def _log_subject(rng: random.Random) -> str:
    return rng.choice([
        "the spike in 503s",
        "the Redis pool exhaustion",
        "the first failed auth",
        "the pod scale event",
        "the cache eviction",
        "the circuit breaker trip",
    ])


# ---------------------------------------------------------------------------
# Random weighted fuzz
# ---------------------------------------------------------------------------

_BLOBS = [
    "short user text.",
    "user text with unicode: 你好 مرحبا שלום 🌱 ↯ – — …",
    "",
    "a" * 512,
    "```python\ndef f(x):\n    return x + 1\n```",
    "{\"nested\": {\"array\": [1, 2, 3], \"unicode\": \"π\"}}",
    "\n".join(f"entry-{i}: lorem ipsum dolor sit amet" for i in range(40)),
    "ERROR: Traceback (most recent call last):\n  File ...\n    raise ValueError(x)",
]


def random_fuzz(
    *, seed: int = 0, target_turns: int = 80,
    enable_rollback: bool = True,
    enable_pins: bool = True,
    enable_docs: bool = True,
    enable_tool_calls: bool = True,
) -> List[Step]:
    """Weighted random turn stream. State-conditional probabilities."""
    rng = random.Random(seed)
    turns: List[Turn] = []
    cid = 0
    live_len_est = 1   # system prompt

    def pick(options: Sequence[str], weights: Sequence[int]) -> str:
        return rng.choices(options, weights=weights, k=1)[0]

    for _ in range(target_turns):
        options: List[str] = ["user", "assistant"]
        weights: List[int] = [40, 25]
        if enable_tool_calls and live_len_est > 5:
            options.append("tool")
            weights.append(20)
        if enable_rollback and live_len_est > 15:
            options.append("rollback")
            weights.append(5)
        if enable_pins and live_len_est > 10:
            options.append("pin")
            weights.append(5)
        if enable_docs and live_len_est > 10:
            options.append("doc_mark")
            weights.append(3)

        action = pick(options, weights)

        if action == "user":
            turns.append(Turn([AddUser(rng.choice(_BLOBS) or "small")]))
            live_len_est += 1
        elif action == "assistant":
            turns.append(Turn([AddAssistant(rng.choice(_BLOBS) or "small")]))
            live_len_est += 1
        elif action == "tool":
            cid += 1
            size = rng.choice([5, 50, 500])
            turns.append(Turn([AddToolCall(
                tool_name=rng.choice(["terminal", "read_file", "memory_search"]),
                arguments={"x": rng.random()},
                result=("blob line\n" * size),
                call_id=f"r{cid:04d}",
            )]))
            live_len_est += 2   # assistant + tool
        elif action == "rollback":
            n = rng.randint(1, min(5, max(1, live_len_est - 3)))
            turns.append(Turn([Rollback(n=n)]))
            live_len_est = max(1, live_len_est - n)
        elif action == "pin":
            # Pin a small recent range. Absolute indices: the live list
            # is at least ``live_len_est`` long; pin the last 3-5.
            width = rng.randint(2, 5)
            end = live_len_est - 1
            start = max(1, end - width)
            turns.append(Turn([MarkPin(
                start_index=start, end_index=end,
                kind="pin", reason="random",
            )]))
        elif action == "doc_mark":
            width = rng.randint(2, 6)
            end = live_len_est - 1
            start = max(1, end - width)
            kind = rng.choice(["doc_chronological", "doc_monolithic"])
            turns.append(Turn([MarkPin(
                start_index=start, end_index=end,
                kind=kind, reason="random doc",
                meta={"title": f"random-{kind}"},
            )]))

    return turns

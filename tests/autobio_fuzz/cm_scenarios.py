"""CM-focused scenarios.

These exercise the Context Manager as a real in-dialogue agent, not
just structurally. Each scenario constructs a state where CM has
something interesting to do — recall from a compressed memory, make a
proactive observation, mediate a multi-turn negotiation — and the
harness captures the full CM transcript (request, dashboard, raw
reply, ops + results) for human review.

Quality is the point here, not structure. Read the report's "CM
dialogues" section and judge whether CM behaves like a useful
collaborator: are the replies specific, on-topic, consistent with the
dashboard, free of contradiction-with-its-own-ops?
"""

from __future__ import annotations

from typing import List

from .steps import (
    AddAssistant, AddToolCall, AddUser, CallCM,
    MarkPin, Rollback, Step, TriggerCompress, TriggerProactiveCheckin,
    Turn,
)


def cm_scenario_recall_after_compression() -> List[Step]:
    """Build up enough conversation that a chunk gets folded into L1,
    then ask CM to recall a specific fact. CM should SEARCH/UNCOMPRESS
    the L1 and answer concretely — not say "I don't have that
    information" before searching."""
    turns: List[Step] = [Turn([AddUser(
        "Setting up a rate-limiting service. Let's design it together."
    )])]
    # Seed a few specific facts CM should be able to recover from a summary.
    turns.append(Turn([AddAssistant(
        "Got it. The key constraints I'd flag upfront:\n"
        "- Backend choice: Redis vs Postgres vs in-memory.\n"
        "- We need persistence across pod restarts.\n"
        "- The /api/rate endpoint is the integration point."
    )]))
    turns.append(Turn([AddUser(
        "Redis seems right for persistence + token-bucket support. "
        "Let's go with Redis. " + "Specifically: pool size 50, "
        "key prefix 'rl:', TTL 300 seconds for buckets. " + "x" * 800
    )]))
    turns.append(Turn([AddAssistant(
        "Agreed on Redis. Pool=50, prefix='rl:', TTL=300s noted. "
        "Next concern: what's the rate limit policy itself? Per-user, "
        "per-IP, per-API-key? And what numbers? " + "y" * 800
    )]))
    turns.append(Turn([AddUser(
        "Per-API-key, 1000 req/min sustained, 5000 burst. " + "z" * 800
    )]))
    turns.append(Turn([AddAssistant(
        "Locked: per-API-key, 1000/min sustained, 5000 burst. "
        "I'll write that into config.yaml under rate_limit. " + "w" * 800
    )]))
    # Filler to push the early facts into a chunk that gets folded.
    for i in range(40):
        turns.append(Turn([AddUser(
            f"Question {i}: tangential follow-up about {_topic(i)}. " + "p" * 800
        )]))
        turns.append(Turn([AddAssistant(
            f"On {_topic(i)}: response. " + "q" * 800
        )]))
    # Force compression to land an L1.
    turns.append(Turn([TriggerCompress()]))
    # Now ask CM to recall the early facts.
    turns.append(Turn([CallCM(
        "What did we decide for the rate-limiting backend? Specifically: "
        "what are the pool size, key prefix, TTL, and per-API-key limits?"
    )]))
    return turns


def cm_scenario_proactive_after_quiet_stretch() -> List[Step]:
    """Long quiet stretch with stable context — CM has no urgent work
    but should produce a coherent metacognitive observation. INJECT vs
    DEFER both valid; the test is whether the OUTPUT reads like a real
    agent thinking, not boilerplate."""
    turns: List[Step] = [Turn([AddUser(
        "Walking through the auth flow refactor. Let me explain what's "
        "in scope: token issuer normalization, audience claim cleanup, "
        "and the cache layer for validated tokens."
    )])]
    for i in range(25):
        turns.append(Turn([AddUser(f"Step {i}: tackle {_subtopic(i)}.")]))
        turns.append(Turn([AddAssistant(
            f"For {_subtopic(i)}: applying the standard pattern, "
            f"verifying against tests/auth/test_{i}.py. " + "a" * 600
        )]))
    # Force the proactive check-in.
    turns.append(Turn([TriggerProactiveCheckin()]))
    return turns


def cm_scenario_mid_chain_recall_unfold() -> List[Step]:
    """Main instance asks CM mid-conversation to unfold a specific
    L1 because it doesn't trust its summary. Tests UNCOMPRESS op and
    CM's ability to negotiate (does it just unfold, or does it warn
    about token cost?)."""
    turns: List[Step] = [Turn([AddUser(
        "Starting an investigation into the Docker build slowness."
    )])]
    # Build up content with enough volume to compress.
    for i in range(50):
        turns.append(Turn([AddUser(
            f"Docker debug step {i}: ran a build, observed timing. " + "d" * 700
        )]))
        turns.append(Turn([AddAssistant(
            f"Timing step {i}: layer N took {i*3}s, suspect {_docker_topic(i)}. "
            + "e" * 700
        )]))
    turns.append(Turn([TriggerCompress()]))
    turns.append(Turn([TriggerCompress()]))    # ensure two L1s
    # Now request unfold of the first L1.
    turns.append(Turn([CallCM(
        "I want the full Docker debugging session back — the early L1. "
        "The summary feels too thin. What's the cost? Can you unfold it?"
    )]))
    return turns


def cm_scenario_halt_and_resume() -> List[Step]:
    """Provoke a halt (we can't easily make K2.6 fail on demand, but
    we can simulate halt by setting the flag manually via a CallCM
    that asks CM to inspect halt status). Mostly tests that CM sees
    halt state in the dashboard and reacts to it."""
    turns: List[Step] = [Turn([AddUser("Working on a deploy plan.")])]
    for i in range(12):
        turns.append(Turn([AddUser(f"Plan item {i}.")]))
        turns.append(Turn([AddAssistant(f"Acknowledged item {i}. " + "h" * 400)]))
    # Ask CM to describe current memory state — specifically about halt.
    turns.append(Turn([CallCM(
        "Are you currently halted on compression? If so, what's the reason "
        "and what should we do about it?"
    )]))
    return turns


def cm_scenario_topic_drift_three_areas() -> List[Step]:
    """Three distinct topics, each long enough to deserve its own L1.
    Then ask CM about a specific topic — does it pick the right L1?"""
    turns: List[Step] = []
    topics = [
        ("auth refactor", "issuer", "normalization", "tests/auth/test_check_token.py"),
        ("rate limiting", "Redis", "token bucket", "config.yaml line 47"),
        ("logging cleanup", "structured", "JSON output", "logger.py:88"),
    ]
    for tid, t in enumerate(topics):
        turns.append(Turn([AddUser(
            f"Switching topics — let's tackle {t[0]} now."
        )]))
        for i in range(15):
            turns.append(Turn([AddUser(
                f"{t[0]} #{i}: thinking about {t[1]} and how it relates "
                f"to {t[2]}. " + chr(ord('a') + tid) * 700
            )]))
            turns.append(Turn([AddAssistant(
                f"For {t[0]}: the {t[1]} angle is what {t[3]} addresses. "
                + chr(ord('A') + tid) * 700
            )]))
        turns.append(Turn([TriggerCompress()]))
    # Now ask CM about the middle topic specifically.
    turns.append(Turn([CallCM(
        "What did we land on for rate limiting? Be specific — backend, "
        "config file, and any concrete numbers."
    )]))
    return turns


# ---------------------------------------------------------------------------
# Filler helpers (deterministic from index — no rng for stability)
# ---------------------------------------------------------------------------

def _topic(i: int) -> str:
    return [
        "circuit breaker", "DLQ semantics", "idempotency keys",
        "TLS pinning", "clock skew", "backpressure", "pool sizing",
    ][i % 7]


def _subtopic(i: int) -> str:
    return [
        "issuer normalization", "audience cleanup", "cache invalidation",
        "token rotation", "JWK refresh", "introspection cache",
        "leeway tuning",
    ][i % 7]


def _docker_topic(i: int) -> str:
    return [
        "the COPY layer", "the npm install step", "the multi-stage join",
        "BuildKit cache miss", "the package-lock checksum",
    ][i % 5]


CM_SCENARIOS = [
    ("cm_recall_after_compression",       cm_scenario_recall_after_compression),
    ("cm_proactive_after_quiet_stretch",  cm_scenario_proactive_after_quiet_stretch),
    ("cm_mid_chain_recall_unfold",        cm_scenario_mid_chain_recall_unfold),
    ("cm_halt_and_resume",                cm_scenario_halt_and_resume),
    ("cm_topic_drift_three_areas",        cm_scenario_topic_drift_three_areas),
]

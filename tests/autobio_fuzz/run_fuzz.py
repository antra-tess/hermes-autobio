"""Entry point for the autobio fuzz harness.

Usage:

    OPENROUTER_API_KEY=sk-or-... \\
        python -m tests.plugins.autobio_fuzz.run_fuzz

Flags:

    --only smoke|curated|scripted|random|all      (default: smoke)
    --model moonshotai/kimi-k2.6                  (override model)
    --seed 0                                       (random fuzz seed)
    --scenarios N                                  (for random mode)
    --turns N                                      (for scripted/random)
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.plugins.autobio_fuzz.cm_scenarios import CM_SCENARIOS
from tests.plugins.autobio_fuzz.generators import random_fuzz, scripted_narrative
from tests.plugins.autobio_fuzz.harness import HarnessConfig, run_many
from tests.plugins.autobio_fuzz.scenarios import ALL_SCENARIOS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="smoke",
                    choices=["smoke", "curated", "scripted", "random", "cm",
                             "replay", "all"])
    ap.add_argument("--replay-jsonl", default=None,
                    help="Path to a Claude Code session JSONL to replay.")
    ap.add_argument("--replay-max-records", type=int, default=None,
                    help="Cap the number of transcript records loaded.")
    ap.add_argument("--model", default="moonshotai/kimi-k2.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", type=int, default=3)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--tail-tokens", type=int, default=15_000)
    ap.add_argument("--chunk-tokens", type=int, default=2_000)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    cfg = HarnessConfig(
        tail_tokens=args.tail_tokens,
        chunk_tokens=args.chunk_tokens,
        model=args.model,
    )

    suites = []
    if args.only in ("smoke",):
        suites.append(("smoke_debug_thread",
                       scripted_narrative("debug_thread", target_turns=args.turns)))
    if args.only in ("scripted", "all"):
        suites.append(("debug_thread",
                       scripted_narrative("debug_thread", target_turns=args.turns)))
        suites.append(("research_thread",
                       scripted_narrative("research_thread", target_turns=args.turns)))
        suites.append(("log_reading_thread",
                       scripted_narrative("log_reading_thread", target_turns=args.turns)))
    if args.only in ("curated", "all"):
        for name, fn in ALL_SCENARIOS:
            suites.append((name, fn()))
    if args.only in ("cm", "all"):
        for name, fn in CM_SCENARIOS:
            suites.append((name, fn()))
    if args.only in ("random", "all"):
        for i in range(args.scenarios):
            suites.append((f"random_seed{args.seed + i}",
                           random_fuzz(seed=args.seed + i, target_turns=args.turns)))
    if args.only == "replay" or (args.only == "all" and args.replay_jsonl):
        if not args.replay_jsonl:
            print("ERROR: --replay-jsonl PATH is required for replay mode",
                  file=sys.stderr)
            return 2
        from tests.plugins.autobio_fuzz.replay import load_claude_jsonl
        steps = load_claude_jsonl(
            args.replay_jsonl, max_records=args.replay_max_records,
        )
        suites.append((
            f"replay_{os.path.basename(args.replay_jsonl).replace('.jsonl', '')}",
            steps,
        ))

    results = run_many(suites, config=cfg)

    total_err = sum(r.errors for r in results)
    total_warn = sum(r.warnings for r in results)
    crashes = [r.scenario_name for r in results if r.crashed]
    print()
    print(f"=== fuzz summary ===")
    print(f"  scenarios: {len(results)}")
    print(f"  errors:    {total_err}")
    print(f"  warnings:  {total_warn}")
    print(f"  crashes:   {len(crashes)} {crashes}")
    for r in results:
        marker = "FAIL" if r.errors or r.crashed else "ok  "
        print(f"  {marker}  {r.scenario_name:40s} err={r.errors} warn={r.warnings} "
              f"transitions={len(r.transitions)}")
    return 0 if total_err == 0 and not crashes else 1


if __name__ == "__main__":
    raise SystemExit(main())

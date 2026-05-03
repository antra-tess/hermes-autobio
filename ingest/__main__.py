"""CLI entry point for archive ingestion.

Usage:

    python -m plugins.autobio.ingest \\
        --source claude_code \\
        --path ~/.claude/projects/<slug>/<session>.jsonl \\
        --hermes-home ~/.hermes \\
        --session-id new_session_name

Adds messages from the source transcript into the autobio archive at
``<hermes-home>/memory/autobio/<session-id>/store.db``. Idempotent —
re-running on the same source skips duplicates.

After seeding, point the engine at the same session-id and it'll see
the seeded archive on the first ``compress()`` call.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..store import close_store, get_store
from .claude_code import ClaudeCodeAdapter
from .seed import seed_archive

_ADAPTERS = {
    "claude_code": ClaudeCodeAdapter,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed an autobio archive from a transcript.")
    ap.add_argument("--source", required=True, choices=sorted(_ADAPTERS),
                    help="Source format.")
    ap.add_argument("--path", required=True, type=Path,
                    help="Path to the source file.")
    ap.add_argument("--hermes-home", default=str(Path.home() / ".hermes"),
                    help="Hermes home dir (default: ~/.hermes)")
    ap.add_argument("--session-id", required=True,
                    help="Destination session id (creates a new archive subdir).")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap on records read from the source.")
    ap.add_argument("--include-system-reminders", action="store_true",
                    default=True)
    ap.add_argument("--no-system-reminders", action="store_false",
                    dest="include_system_reminders")
    ap.add_argument("--raise-on-failure", action="store_true",
                    help="Halt on the first upsert failure (default: continue).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.path.exists():
        print(f"ERROR: source path not found: {args.path}", file=sys.stderr)
        return 2

    adapter_cls = _ADAPTERS[args.source]
    adapter_kwargs = {
        "path": args.path,
        "max_records": args.max_records,
    }
    if args.source == "claude_code":
        adapter_kwargs["include_system_reminders"] = args.include_system_reminders

    adapter = adapter_cls(**adapter_kwargs)

    store = get_store(args.hermes_home, args.session_id)
    try:
        result = seed_archive(
            store, adapter, raise_on_failure=args.raise_on_failure,
        )
        print(f"== ingest result ==")
        print(f"  source:     {args.source} {args.path}")
        print(f"  session:    {args.session_id}")
        print(f"  inserted:   {result.inserted}")
        print(f"  skipped:    {result.skipped} (already in archive)")
        print(f"  failed:     {result.failed}")
        if result.sample_failures:
            print(f"  failure samples:")
            for fs in result.sample_failures:
                print(f"    - {fs}")
        print(f"  last turn:  {result.last_turn}")
        print(f"  last pos:   {result.last_position}")
        return 0 if result.failed == 0 else 1
    finally:
        close_store(args.hermes_home, args.session_id)


if __name__ == "__main__":
    raise SystemExit(main())

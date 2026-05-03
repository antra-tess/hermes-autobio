"""CLI entry: ``python -m plugins.autobio.viewer --hermes-home PATH``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .app import create_app


def main() -> int:
    ap = argparse.ArgumentParser(description="Autobio viewer (Flask).")
    ap.add_argument("--hermes-home", default=str(Path.home() / ".hermes"),
                    help="Hermes home dir (default: ~/.hermes).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debug", action="store_true",
                    help="Flask debug mode (auto-reload).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    home = Path(args.hermes_home).expanduser()
    if not home.exists():
        print(f"warning: hermes_home {home} does not exist", file=sys.stderr)

    app = create_app(home)
    print(f"autobio viewer listening on http://{args.host}:{args.port}")
    print(f"hermes_home = {home}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

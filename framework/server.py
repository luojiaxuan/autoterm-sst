"""Framework entry point: ``python -m framework.server``.

Builds the agent router from config and serves the thin transport layer over
uvicorn. This replaces the three standalone servers in ``serve/`` as the single
demo entry point; the existing ``serve/static`` UI is served unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from framework.app import create_app
from framework.config import build_router


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoTerm-SST streaming SST framework")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated agent_type list to load (default: env RASST_FRAMEWORK_AGENTS or 'InfiniSST,RASST').",
    )
    parser.add_argument(
        "--default-agent",
        default=None,
        help="agent_type used when /init omits or sends an unknown agent_type.",
    )
    parser.add_argument("--log-level", default=os.environ.get("RASST_LOG_LEVEL", "info"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = build_router(agents=args.agents, default_agent=args.default_agent)
    app = create_app(router)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

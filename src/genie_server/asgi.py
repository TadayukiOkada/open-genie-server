"""ASGI entry point for running under uvicorn directly:

    uvicorn genie_server.asgi:app --host 0.0.0.0 --port 8080 --workers 1

Always keep --workers at 1: each slot's GenieDialog handle is process-global
state; multiple worker processes would fight over separate NPU contexts.

The config path can be overridden with the GENIE_SERVER_CONFIG environment
variable (default: ./env_config.json).
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from .bootstrap import build_app  # noqa: E402

app = build_app(os.environ.get("GENIE_SERVER_CONFIG", "env_config.json"))

#!/usr/bin/env python3
"""Serves the genie-server FastAPI app against the fake SDK on localhost.

For developing/validating the integration test harness without a device:

    python3 fake_server.py --port 18080 &
    python3 run_integration_tests.py --base-url http://127.0.0.1:18080
"""

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    from conftest import build_state
    from genie_server.app import create_app

    state = build_state(Path(tempfile.mkdtemp(prefix="genie-fake-")))
    app = create_app(state)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

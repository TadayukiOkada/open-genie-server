#!/usr/bin/env python3
"""genie-server launcher — OpenAI-compatible REST API for Qualcomm Genie.

    python3 genie-server.py [--config env_config.json] [--host H] [--port P]

This file is a shim. The command line, the startup wiring and the lm_eval
recipes all live in genie_server/cli.py; installing the package also gives you
the same thing as a `genie-server` command.

The shim is kept because it is what the on-device deployment starts, and what
`pgrep -f '[g]enie-server.py'` matches when you need the PID to stop it. It
finds the package in all three layouts we actually use:

  - installed (`pip install .` or `pip install -e .`) — the normal case;
  - a source checkout with no install at all — the sibling `src/` is added to
    sys.path below, so the development loop needs no install step;
  - a device deployment where `genie_server/` was copied next to this file
    (`/home/root/genie_server/`) — Python already puts this file's directory
    on sys.path, so the first import succeeds.
"""

import sys
from pathlib import Path

try:
    from genie_server.cli import main
except ImportError:
    _src = Path(__file__).resolve().parent / "src"
    if not (_src / "genie_server" / "cli.py").is_file():
        raise
    sys.path.insert(0, str(_src))
    from genie_server.cli import main


if __name__ == "__main__":
    main()

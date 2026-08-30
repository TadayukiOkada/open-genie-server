"""Command-line entry point — `genie-server`, and the `genie-server.py` shim.

The implementation lives in the sibling modules (see the package docstring);
this one only parses the command line and hands the FastAPI app built by
genie_server.bootstrap to uvicorn.

Usage:
    genie-server [--config env_config.json] [--host H] [--port P]
    python3 genie-server.py [--config env_config.json] [--host H] [--port P]

Both spellings run this same main(). The console script exists once the
package is installed; the repository-root launcher works either way, and is
what the on-device deployment starts (it is also what `pgrep -f
'[g]enie-server.py'` matches).

lm_eval usage:
  # Generation tasks (gsm8k, ...)
  lm_eval --model local-completions \
    --model_args model=genie-local,base_url=http://<ip>:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
    --tasks gsm8k --batch_size 1

  # Chat tasks
  lm_eval --model local-chat-completions \
    --model_args model=genie-local,base_url=http://<ip>:8080/v1,\
tokenizer_backend=huggingface,tokenizer=<hf_model>,max_tokens=512,num_concurrent=1 \
    --tasks mmlu_generative --apply_chat_template --batch_size 1

  # To reach the SECOND slot on a dual-NSP box, pass its active_model_id
  # (== that slot's model directory name) as model=... instead of genie-local.
"""

import argparse
import logging
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genie-server",
        description="OpenAI-compatible REST API server for Qualcomm Genie")
    parser.add_argument("--config", default="env_config.json",
                        help="path to env_config.json (default: ./env_config.json)")
    parser.add_argument("--host", default=None, help="bind address (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="listen port (overrides config)")
    parser.add_argument("--version", action="version",
                        version=f"genie-server {__version__}")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("genie-server")

    from .app import create_app
    from .bootstrap import build_state

    try:
        state = build_state(args.config)
    except FileNotFoundError as e:
        logger.error(f"Startup failed — file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

    app = create_app(state)

    import uvicorn
    uvicorn.run(app,
                host=args.host or state.config.host,
                port=args.port or state.config.port)


if __name__ == "__main__":
    main()

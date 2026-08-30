"""Startup wiring: config -> environment -> libGenie -> slots -> FastAPI app.

Shared by the command line (genie_server.cli, reached as `genie-server` or
`python3 genie-server.py`) and the ASGI entry point (genie_server.asgi).
Raises on failure — the caller decides whether that is fatal.
"""

import logging

from fastapi import FastAPI

from .app import ServerState, create_app
from .config import ServerConfig, load_config
from .prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


def build_state(config_path: str = "env_config.json") -> ServerState:
    config: ServerConfig = load_config(config_path)

    # Environment variables must be in place BEFORE libGenie.so is loaded.
    config.apply_process_env()

    from .capi import GenieLib
    so_path = config.resolved_genie_lib_path()
    lib = GenieLib.load(so_path)
    logger.info(f"libGenie.so loaded from: {so_path}")

    from . import vlm
    from .slots import SlotManager

    logger.info("Initializing Genie Core Components...")
    manager = SlotManager(config, lib)
    # Creation order matters when both kinds of slot are configured; see
    # ServerConfig.slot_load_order and the note in vlm.create_vlm_slots().
    if config.slot_load_order == "text-first":
        logger.info("Slot load order: text slots first, then VLM slots.")
        manager.load_all()
        # Mandatory between the two: the text dialogs just set libGenie's
        # process-global positional-encoding validator flags, which would
        # otherwise reject every VLM text-generator node config.
        lib.reset_dialog_validator_flags()
        manager.vlm_slots = vlm.create_vlm_slots(config, lib.cdll)
    else:
        manager.vlm_slots = vlm.create_vlm_slots(config, lib.cdll)
        manager.load_all()

    return ServerState(
        config=config,
        lib=lib,
        manager=manager,
        prefix_cache=PrefixCache(config.prefix_cache_dir),
    )


def build_app(config_path: str = "env_config.json") -> FastAPI:
    return create_app(build_state(config_path))

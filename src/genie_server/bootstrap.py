"""Startup wiring: config -> environment -> libGenie -> slots -> FastAPI app.

Shared by the command line (genie_server.cli, reached as `genie-server` or
`python3 genie-server.py`) and the ASGI entry point (genie_server.asgi).
Raises on failure — the caller decides whether that is fatal.
"""

import logging
import os

from fastapi import FastAPI

from .app import ServerState, create_app
from .config import ServerConfig, load_config
from .prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


def build_state(config_path: str = "env_config.json") -> ServerState:
    config: ServerConfig = load_config(config_path)

    # Environment variables must be in place BEFORE libGenie.so is loaded.
    logger.info(f"Target platform: {config.platform} "
                f"(TARGET_PLATFORM={config.target_platform})")
    config.apply_process_env()

    from .capi import GenieLib
    so_path = config.resolved_genie_lib_path()
    lib = GenieLib.load(so_path)
    logger.info(f"libGenie.so loaded from: {_mapped_path(so_path)}")

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
        manager.vlm_slots = vlm.create_vlm_slots(config, lib.cdll,
                                                 manager.log_handle)
    else:
        manager.vlm_slots = vlm.create_vlm_slots(config, lib.cdll,
                                                 manager.log_handle)
        manager.load_all()

    return ServerState(
        config=config,
        lib=lib,
        manager=manager,
        prefix_cache=PrefixCache(config.prefix_cache_dir),
    )


def _mapped_path(so_path: str) -> str:
    """The file the loader actually mapped for so_path. A bare name such as
    "libGenie.so" resolves through LD_LIBRARY_PATH and the system cache, and
    the name alone does not say which copy won."""
    name = os.path.basename(so_path)
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                path = line.split(maxsplit=5)[-1].strip()
                if os.path.basename(path) == name:
                    return path if path == so_path else f"{path} (requested {so_path})"
    except OSError:
        pass
    return so_path


def build_app(config_path: str = "env_config.json") -> FastAPI:
    return create_app(build_state(config_path))

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# src layout: the package is importable from a plain checkout, with no
# install step, exactly as the genie-server.py launcher arranges it.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from genie_server.app import ServerState, create_app  # noqa: E402
from genie_server.config import ServerConfig, SlotSpec  # noqa: E402
from genie_server.prefix_cache import PrefixCache  # noqa: E402
from genie_server.slots import Slot, SlotManager  # noqa: E402
from fake_genie import FakeGenieLib, FakeTokenizer  # noqa: E402


def build_state(tmp_path, template: str = "chatml") -> ServerState:
    model_root = Path("/models/qwen3-test")
    cfg = ServerConfig(
        sdk_root="/nonexistent",
        prefix_cache_dir=str(tmp_path / "prefix_cache"),
        inference_timeout_s=10.0,
        text_slots=(SlotSpec(name="default", device_id=None,
                             model_root=model_root),),
    )
    lib = FakeGenieLib()
    manager = SlotManager(cfg, lib)
    slot = Slot(name="default", device_id=None, model_root=model_root)
    slot.handle = lib.create_dialog(b"{}")
    slot.dialog_cfg = {"context": {"size": 4096}}
    slot.chat_template = template
    slot.tokenizer = FakeTokenizer()
    lib.tokenizer = slot.tokenizer  # the fake plans token ids with it
    manager.slots = [slot]
    manager.status[slot.name] = {"phase": "idle", "detail": ""}
    manager._by_name = {slot.name: slot}
    manager.reindex()
    return ServerState(config=cfg, lib=lib, manager=manager,
                       prefix_cache=PrefixCache(cfg.prefix_cache_dir))


@pytest.fixture
def state(tmp_path):
    return build_state(tmp_path)


@pytest.fixture
def client(state):
    return TestClient(create_app(state))


@pytest.fixture
def client_recovery_on(state):
    """A client with TOOL_CALL_RECOVERY on, which is not the default.

    The default is off so that a bundle which mangles its own <tool_call>
    marker measures as what it is. The recovery still has to work when asked
    for, so the tests that cover it opt in here rather than relying on a
    default that says something else.
    """
    import dataclasses
    state.config = dataclasses.replace(state.config, tool_call_recovery=True)
    return TestClient(create_app(state))

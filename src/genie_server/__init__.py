"""genie-server — OpenAI-compatible REST API server for Qualcomm Genie (libGenie.so).

Package layout:
    config.py        env_config.json parsing and process-environment setup
    capi.py          ctypes bindings for the GenieDialog C API + SDK constants
    templates.py     chat-template rendering (chatml / llama3 / llama2)
    tools.py         OpenAI `tools` (function calling) <-> Hermes prompt format
    slots.py         Slot / VLMSlot / SlotManager (model loading, routing, hot-swap)
    prefix_cache.py  on-disk KV-cache snapshots for system-prompt prefixes
    engine.py        the generation engine (lock, watchdog, prefix cache, SDK params)
    vlm.py           multimodal (image) request parsing + GenieNode pipeline driving
    genie_node.py    ctypes bindings for the GenieNode/GeniePipeline composable API
    vlm_specs.py     per-VLM-model specs (preprocessing, topology, prompt template)
    logprobs.py      token logprobs via the SDK's custom-sampler hook
    protocol.py      OpenAI wire-format builders and error shapes
    app.py           FastAPI application factory (all HTTP routes)
    cli.py           command-line entry point (argument parsing -> uvicorn)

Run it either way — both call genie_server.cli.main():
    genie-server                  (console script, once the package is installed)
    python3 genie-server.py       (repository-root launcher; needs no install)
"""

__version__ = "1.0.0"

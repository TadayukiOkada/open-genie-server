"""FastAPI application factory — all HTTP routes.

Endpoints (OpenAI-compatible core, also registered without the /v1 prefix):
  POST /v1/completions          — raw text completion (lm_eval local-completions)
  POST /v1/chat/completions     — chat completion (streaming, tools, VLM routing)
  GET  /v1/models               — lists all loaded slots' models
  GET  /v1/models/{model_id}
  GET  /health, /v1/health      — liveness probe

Server management:
  POST /v1/models/switch        — hot-swap one slot's model
  GET  /v1/server/idle          — wait until a slot's lock is free (?slot=name)
  GET  /v1/server/status        — per-slot phase, context occupancy, model/LoRA
  GET  /v1/server/profile       — SDK-side KPIs for the last query (GENIE_PROFILE)
  GET/POST /v1/server/performance_policy
  GET  /v1/prefix/cache         — list KV cache entries
  DELETE /v1/prefix/cache/{key}
  POST /v1/prefix/warmup        — pre-populate the prefix KV cache
  POST /v1/lora/apply | /v1/lora/strength | /v1/lora/release
  GET  /v1/lora/current
"""

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import engine, logprobs as logprobs_mod, protocol, templates, \
    vlm
from .capi import (GenieLib, PERFORMANCE_POLICIES, PERFORMANCE_POLICY_NAMES,
                   STATUS_SUCCESS)
from .config import ServerConfig, resolve_model_path
from .engine import GenParams, Generation, QueryPlan, SlotChangedError
from .logprobs import LogprobsCollector
from .prefix_cache import PrefixCache
from .protocol import InvalidRequestError, openai_error, read_json_body, sse
from .slots import SlotManager, canonical_engine_role, lora_alpha_names

logger = logging.getLogger(__name__)

# Historical fixed model id, accepted (and echoed back) from clients such as
# lm_eval, which always sends one fixed placeholder for every request.
KNOWN_MODEL_ID = "genie-local"


@dataclass
class ServerState:
    config: ServerConfig
    lib: GenieLib
    manager: SlotManager
    prefix_cache: PrefixCache
    # Runtime switch for prompt scoring (echo+logprobs teacher forcing).
    # Initialized from config (PROMPT_LOGPROBS); toggled at runtime via
    # POST /v1/server/prompt_logprobs. Deliberately explicit: one scoring
    # request occupies its slot for len(prompt)/decode-rate seconds.
    prompt_logprobs_enabled: bool = False

    def __post_init__(self):
        self.prompt_logprobs_enabled = self.config.prompt_logprobs


# ---------------------------------------------------------------- body parsing

# The SDK reads the seed with std::stoi, and the logprobs sampler hands it to
# numpy, which refuses a negative one.
_MAX_SEED = 2**31 - 1


def _number(body: dict, key: str, *, lo: float | None = None,
            hi: float | None = None, integer: bool = False):
    """body[key] as a validated number, or None when it is absent (or null).

    Checked here, before anything reaches a worker thread: a string there
    used to be a 500 (float("hot")) or, worse, pass (top_k 1.5 was cut to 1
    by int(), turning a sampled request greedy without a word). A bool is not
    a number, whatever Python says; a float with an integral value (2.0, as
    some clients serialize every number) counts as an integer."""
    value = body.get(key)
    if value is None:
        return None
    ok = (not isinstance(value, bool) and isinstance(value, (int, float))
          and math.isfinite(value))
    if ok and integer:
        ok = float(value).is_integer()
        value = int(value) if ok else value
    if ok and ((lo is not None and value < lo) or (hi is not None and value > hi)):
        ok = False
    if not ok:
        kind = "an integer" if integer else "a number"
        if lo is not None and hi is not None:
            rng = f" between {lo:g} and {hi:g}"
        elif lo is not None:
            rng = f" >= {lo:g}"
        else:
            rng = ""
        raise InvalidRequestError(f"{key} must be {kind}{rng}, got {value!r}",
                                  key)
    return value


def _flag(body: dict, key: str, default: bool = False) -> bool:
    """A boolean field. bool() alone read the string "false" as true."""
    value = body.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise InvalidRequestError(f"{key} must be true or false, got {value!r}",
                                  key)
    return value


def _parse_gen_params(body: dict, allow_stop: bool = True) -> GenParams:
    """Extracts/validates OpenAI generation parameters from a request body.
    max_completion_tokens (current OpenAI name) takes priority over the
    legacy max_tokens, mirroring OpenAI's own deprecation behavior."""
    # 0 is accepted here because lm_eval's loglikelihood (prompt scoring)
    # requests legitimately send max_tokens=0; the non-scoring paths reject
    # it explicitly (_require_positive_max_tokens).
    # _number, as for every other count: true was read as 1, while 8.0 --
    # how some clients serialize every number -- was refused.
    max_tokens = _number(body, "max_completion_tokens", lo=0, integer=True)
    if max_tokens is None:
        max_tokens = _number(body, "max_tokens", lo=0, integer=True)

    stop = body.get("stop") if allow_stop else None
    if stop is None:
        stop_list: list[str] = []
    elif isinstance(stop, str):
        stop_list = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        stop_list = list(stop)
    else:
        raise InvalidRequestError("stop must be a string or an array of strings", "stop")

    top_k = body.get("top_k")
    if (isinstance(top_k, (int, float)) and not isinstance(top_k, bool)
            and top_k < 0 and top_k != -1):
        raise InvalidRequestError(
            f"top_k must be an integer >= 0 (0, or vLLM's -1, means no top-k "
            f"limit), got {top_k!r}", "top_k")
    top_k = _number(body, "top_k", lo=-1, integer=True)
    if top_k == -1:
        # vLLM's spelling of "no top-k limit", which is 0 here: the SDK reads
        # top-k as unsigned. Same meaning, so nothing about the model or the
        # SDK is hidden by accepting it, and a client written for vLLM works.
        top_k = 0
    return GenParams(
        max_tokens=max_tokens,
        stop=stop_list,
        temperature=_number(body, "temperature", lo=0),
        top_p=_number(body, "top_p", lo=0, hi=1),
        top_k=top_k,
        seed=_number(body, "seed", lo=0, hi=_MAX_SEED, integer=True),
    )


# tool_choice values this server can honor. "auto" (and an absent field)
# means "inject the tools and let the model decide"; "none" suppresses the
# injection. OpenAI's "required" and the {"type":"function", ...} form both
# *guarantee* a call, which needs constrained decoding we do not implement —
# so they are rejected rather than silently degraded to "auto", which would
# hand the caller ordinary prose where their code expects tool_calls.
_SUPPORTED_TOOL_CHOICE = (None, "auto", "none")


def _reject_unsupported(body: dict, endpoint: str) -> None:
    if (_number(body, "n", lo=1, integer=True) or 1) > 1:
        raise InvalidRequestError(
            "n > 1 is not supported by this server (one NPU handle per slot "
            "serves one completion at a time).", "n")
    if endpoint == "chat":
        choice = body.get("tool_choice")
        if choice not in _SUPPORTED_TOOL_CHOICE:
            named = ""
            if isinstance(choice, dict):
                named = (choice.get("function") or {}).get("name") or ""
            raise InvalidRequestError(
                'tool_choice supports only "auto" and "none" on this server. '
                + (f'Forcing a specific function ({named!r}) ' if named
                   else f'{choice!r} ')
                + "requires constrained decoding, which is not implemented; "
                  'use "auto" and check whether the model returned '
                  "tool_calls.", "tool_choice")
    if endpoint == "completions" and body.get("suffix"):
        raise InvalidRequestError("suffix (fill-in-the-middle) is not supported", "suffix")
    if endpoint == "completions" and (
            _number(body, "best_of", lo=1, integer=True) or 1) > 1:
        raise InvalidRequestError("best_of > 1 is not supported", "best_of")


def _include_usage(body: dict) -> bool:
    """stream_options.include_usage. Called where the request is parsed,
    before a generation starts: a 400 raised once the stream is being set up
    would leave that generation holding the slot."""
    options = body.get("stream_options")
    if options is None:
        return False
    if not isinstance(options, dict):
        raise InvalidRequestError("stream_options must be an object",
                                  "stream_options")
    value = options.get("include_usage")
    if value is None:
        return False
    if not isinstance(value, bool):
        # bool("false") is true: a client asking for no usage chunk got one.
        raise InvalidRequestError(
            f"stream_options.include_usage must be true or false, got {value!r}",
            "stream_options.include_usage")
    return value


_MAX_TOP_LOGPROBS = 20
# Wall-clock budget per teacher-forced token when scaling the watchdog for
# prompt scoring — generous enough for ~2 tok/s decode targets.
_SCORING_SECONDS_PER_TOKEN = 0.5


def _require_positive_max_tokens(params: GenParams) -> None:
    if params.max_tokens == 0:
        raise InvalidRequestError(
            "max_tokens=0 is only valid for prompt scoring (echo+logprobs); "
            "use max_tokens >= 1 to generate.", "max_tokens")


def _require_logprobs_support(slot) -> None:
    if not logprobs_mod.LOGPROBS_AVAILABLE:
        raise InvalidRequestError(
            "logprobs require numpy on the server (pip install numpy).",
            "logprobs")
    if slot.tokenizer is None:
        raise InvalidRequestError(
            "logprobs require the model tokenizer, which is not loaded on "
            "this server (install 'tokenizers' and ensure genie_config.json "
            "points at tokenizer.json).", "logprobs")
    if getattr(slot, "logits_callback_unsupported", False):
        raise _logprobs_unsupported_error()


def _logprobs_unsupported_error() -> InvalidRequestError:
    return InvalidRequestError(
        "logprobs are not supported by this QAIRT runtime: the logits "
        "callback was not invoked during generation.",
        "logprobs", code="logprobs_not_supported")


def _completions_top_n(body: dict) -> int | None:
    """OpenAI completions `logprobs`: an int = number of top alternatives
    (the chosen token's logprob is always included). None/0/False = off."""
    logprobs = body.get("logprobs")
    if not logprobs:
        return None
    if logprobs is True:
        return 1
    if not isinstance(logprobs, int) or logprobs < 0 or logprobs > _MAX_TOP_LOGPROBS:
        raise InvalidRequestError(
            f"logprobs must be an integer between 0 and {_MAX_TOP_LOGPROBS}",
            "logprobs")
    return logprobs


def _chat_top_n(body: dict) -> int | None:
    """OpenAI chat `logprobs`: a bool, plus `top_logprobs` (0-20). Returns
    the top-N to record, or None when logprobs are off."""
    if not _flag(body, "logprobs"):
        return None
    # _number, not isinstance(int): true is an int to Python and read as 1.
    top = _number(body, "top_logprobs", lo=0, hi=_MAX_TOP_LOGPROBS, integer=True)
    return 0 if top is None else top


# ---------------------------------------------------------------- streaming

async def _sse_stream(
    request: Request,
    state: ServerState,
    gen: Generation,
    model_name: str,
    make_delta,             # (text) -> chunk dict
    make_final,             # (finish_reason) -> chunk dict
    preamble: list | None = None,
    tool_filter=None,   # a tool_formats stream filter, or None
    include_usage: bool = False,
    prompt_tokens: int = 0,
    usage_object_type: str = "chat.completion.chunk",
    abortable: bool = True,
):
    """Shared SSE generator for chat and text completion streaming."""
    try:
        async for chunk in _sse_body(request, state, gen, model_name, make_delta,
                                     make_final, preamble, tool_filter,
                                     include_usage, prompt_tokens,
                                     usage_object_type, abortable):
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        # This is how a real client disconnect arrives. Starlette's
        # StreamingResponse runs its own listen_for_disconnect task and consumes
        # the receive channel itself, so request.is_disconnected() below never
        # returns True under uvicorn -- it cancels this iterator instead.
        # Without this the generation ran to completion with nobody listening,
        # holding the slot: measured on the board, hanging up 0.4s into an 81s
        # completion made the next request wait 82s. The in-loop check stays for
        # transports that do deliver the message (TestClient does).
        if abortable and not gen.done.is_set():
            logger.info(f"Client disconnected (stream cancelled); ABORT "
                        f"[{gen.request_id}]")
            gen.abort()          # synchronous: safe during GeneratorExit
        elif not abortable and not gen.done.is_set():
            logger.info(f"Client disconnected (stream cancelled) [{gen.request_id}]; "
                        "VLM inference continues server-side (no abort API)")
        raise


async def _sse_body(
    request: Request,
    state: ServerState,
    gen: Generation,
    model_name: str,
    make_delta,
    make_final,
    preamble: list | None,
    tool_filter,        # a tool_formats stream filter, or None
    include_usage: bool,
    prompt_tokens: int,
    usage_object_type: str,
    abortable: bool,
):
    """The stream itself. Split out so _sse_stream can wrap it in the
    cancellation handling above without indenting all of it."""
    for chunk in (preamble or []):
        yield sse(chunk)

    while True:
        if await request.is_disconnected():
            if abortable:
                logger.info(f"Client disconnected; ABORT [{gen.request_id}]")
                gen.abort()
                deadline = time.monotonic() + state.config.abort_drain_timeout_s
                while not gen.done.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            else:
                logger.info(f"Client disconnected [{gen.request_id}]; VLM inference "
                            "continues server-side (no abort API for GenieNode)")
            return

        try:
            token = gen.queue.get_nowait()
        except asyncio.QueueEmpty:
            try:
                token = await asyncio.wait_for(gen.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

        if token is None:
            break
        text = tool_filter.feed(token) if tool_filter else token
        if text:
            yield sse(make_delta(text))

    if tool_filter:
        leftover, tool_calls = tool_filter.finalize()
        if leftover:
            yield sse(make_delta(leftover))
        if tool_calls:
            yield sse(protocol.tool_calls_chunk(gen.request_id, model_name, tool_calls))
            gen.finish_reason = "tool_calls"

    if gen.error:
        # Mid-stream failure: emit an error event (vLLM-style), then close.
        yield sse({"error": {"message": gen.error, "type": "server_error",
                             "param": None, "code": None}})
    else:
        yield sse(make_final(gen.finish_reason))
        if include_usage:
            yield sse(protocol.chat_usage_chunk(
                gen.request_id, model_name, prompt_tokens, gen.completion_tokens,
                usage_object_type))
    yield "data: [DONE]\n\n"


async def _collect_or_raise(gen: Generation, state: ServerState,
                            timeout_s: float | None = None,
                            collector: LogprobsCollector | None = None) -> str:
    """Sync path: waits for the full completion; maps failures to HTTP."""
    try:
        text = await gen.collect_text((timeout_s or state.config.inference_timeout_s) * 2)
    except asyncio.CancelledError:
        gen.abort()
        raise
    except SlotChangedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TimeoutError:
        # Free the slot: without this the worker keeps waiting for the lock,
        # then runs the abandoned generation to its end.
        gen.abort()
        raise HTTPException(
            status_code=504,
            detail=f"Inference timed out on Hexagon NPU [{gen.request_id}]")
    except RuntimeError as e:
        if collector is not None and gen.logprobs_unsupported:
            raise _logprobs_unsupported_error() from None
        raise HTTPException(status_code=500, detail=str(e))
    # The engine stops the query at the first token that arrives without a
    # logits callback. The second test is a backstop for a runtime that emits
    # nothing through the token callback yet still returns partial text.
    if collector is not None and (gen.logprobs_unsupported or (
            gen.completion_tokens and not collector.results)):
        raise _logprobs_unsupported_error()
    return text


# ---------------------------------------------------------------- slot-locked work

class SlotBusyError(HTTPException):
    """The slot's lock was not free in time (503). A type of its own so a
    caller can tell "busy" apart from an HTTPException the work raised."""

    def __init__(self, detail: str):
        super().__init__(status_code=503, detail=detail)


class ClientGoneError(HTTPException):
    """The client disconnected while its request waited for the slot. Nobody
    reads this response; the type exists so the handler stops cleanly."""

    def __init__(self, detail: str):
        super().__init__(status_code=499, detail=detail)


# How often a request waiting for a slot lock checks whether its client is
# still there.
_DISCONNECT_POLL_S = 0.5

# Threads for management calls that wait on a slot lock. Separate from the
# loop's default executor, which /v1/server/idle, the warmup join and media
# decoding share: a switch can wait 600 s for its slot, and a few of those
# queued behind a long generation must not starve the rest.
_SLOT_ADMIN_WORKERS = 8


async def run_with_slot_lock(slot, fn, *, timeout: float, executor=None,
                             request: Request | None = None,
                             require_loaded: bool = True,
                             busy_detail: str | None = None):
    """Runs fn() holding slot.lock, on a worker thread, and returns its result.

    Waiting for the lock (a generation can hold it for inference_timeout_s)
    and the SDK call itself both block, so neither may run on the event loop:
    that would stall every other slot's token delivery and /health.

    - SlotBusyError (503) if the lock is not acquired within `timeout`,
      counted from this call — time spent queued for an executor thread
      counts against it.
    - With `require_loaded`, the slot is checked for a model AFTER the lock
      is acquired. Checking before is not enough: a failed unload_first
      switch can empty the slot while this call waits, and fn() would then
      hand a null handle to the SDK. A switch passes False, since it is how
      an empty slot gets its model back.
    - With `request`, the client is polled while the call waits. If it has
      gone, fn() is not run once the lock is acquired, and ClientGoneError is
      raised. Starlette does not cancel a plain request's handler on
      disconnect, and a thread cannot be cancelled, so without this a switch
      the client gave up on would still happen up to 600 s later. A
      disconnect after fn() has started cannot be undone.
    """
    deadline = time.monotonic() + timeout
    abandoned = threading.Event()

    def _run():
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not slot.lock.acquire(timeout=remaining):
            raise SlotBusyError(
                busy_detail or f"Slot '{slot.name}' busy; could not acquire lock.")
        try:
            if abandoned.is_set():
                logger.info(f"[{slot.name}] Client disconnected while waiting "
                            "for the slot; not running the request.")
                return None
            if require_loaded:
                SlotManager.require_loaded(slot)
            return fn()
        finally:
            slot.lock.release()

    fut = asyncio.get_running_loop().run_in_executor(executor, _run)

    def _abandon():
        abandoned.set()
        # Nobody awaits the future any more; retrieve its outcome so asyncio
        # does not log "exception was never retrieved" for it.
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())

    try:
        if request is None:
            return await fut
        while True:
            done, _ = await asyncio.wait({fut}, timeout=_DISCONNECT_POLL_S)
            if done:
                return fut.result()
            if await request.is_disconnected():
                _abandon()
                raise ClientGoneError(
                    f"Client disconnected while waiting for slot '{slot.name}'.")
    except asyncio.CancelledError:
        _abandon()
        raise


# ---------------------------------------------------------------- app factory

def create_app(state: ServerState) -> FastAPI:
    slot_admin_executor = ThreadPoolExecutor(
        max_workers=_SLOT_ADMIN_WORKERS, thread_name_prefix="slot-admin")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        slot_admin_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Releasing HTP context memory...")
        state.manager.free_all()

    app = FastAPI(title="Genie OpenAI-Compatible Server", lifespan=lifespan)
    app.state.max_request_body_bytes = int(
        state.config.max_request_body_mb * 2**20)

    # Added BEFORE CORS on purpose: Starlette wraps later middleware around
    # earlier ones, so this sits inside CORS and its 500 carries the CORS
    # headers an allowed page needs to read it (see UnexpectedErrorMiddleware).
    app.add_middleware(protocol.UnexpectedErrorMiddleware)

    # CORS: off unless CORS_ALLOW_ORIGINS names the origins a browser page may
    # call from (see ServerConfig.cors_allow_origins). With no middleware a
    # cross-origin page cannot read a reply or pass a preflight; the simple
    # requests a browser sends without asking are refused by read_json_body.
    # (allow_credentials stays False: browsers reject "*" with credentials.)
    if state.config.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(state.config.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    protocol.install_error_handlers(app)

    manager = state.manager
    cfg = state.config

    # ------------------------------------------------------------ health

    @app.get("/health")
    @app.get("/v1/health")
    async def health():
        """Liveness probe (vLLM-compatible shape)."""
        return {"status": "ok"}

    @app.get("/ready")
    @app.get("/v1/ready")
    async def ready():
        """Readiness probe: 200 when every slot holds a model, 503 when one
        does not -- mid-switch, or left empty by a failed unload_first switch,
        after which /health still says ok and every request to that slot
        fails. Startup never shows here: the server does not listen until
        every slot has loaded.

        "Loaded" is all it knows. A slot the stock library has wedged still
        holds its model and still reads as ready: telling a wedge from a slow
        answer means sending a request, and this server does not detect the
        wedge or paper over it (README, principle 3)."""
        slots = [{"name": s.name, "loaded": s.handle is not None}
                 for s in manager.slots]
        slots += [{"name": v.name, "loaded": v.pipeline is not None}
                  for v in manager.vlm_slots]
        not_loaded = [s["name"] for s in slots if not s["loaded"]]
        body = {"status": "not ready" if not_loaded else "ready",
                "slots": slots}
        if not_loaded:
            body["not_loaded"] = not_loaded
            return JSONResponse(status_code=503, content=body)
        return body

    # ------------------------------------------------------------ models

    @app.get("/models")
    @app.get("/v1/models")
    async def list_models():
        """Required by lm_eval, Open WebUI's model picker, and most OpenAI
        clients. Always includes the fixed 'genie-local' placeholder plus
        every slot's real active_model_id — text slots AND VLM slots."""
        ids = ({KNOWN_MODEL_ID}
               | {s.active_model_id for s in manager.slots}
               | {s.active_model_id for s in manager.vlm_slots})
        return {"object": "list",
                "data": [protocol.model_object(i) for i in sorted(ids)]}

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str):
        """Used by LangChain/LiteLLM to validate a model before use. Accepts
        ANY id: the request 'model' field is only ever used for slot routing
        (with a fallback), so there is no fixed allow-list."""
        return protocol.model_object(model_id)

    async def _admin_with_slot_lock(request, slot, fn, timeout, **kwargs):
        """A management call that changes the slot: on the dedicated
        executor, and dropped if its client leaves while it waits."""
        return await run_with_slot_lock(
            slot, fn, timeout=timeout, executor=slot_admin_executor,
            request=request, **kwargs)

    @app.post("/v1/models/switch")
    async def switch_model(request: Request):
        """Hot-swaps the model loaded into one hardware slot (NSP core).
        Body: {"slot": "chat", "model_dir": "<model directory>",
               "config_file": "genie_config.json", "unload_first": true}

        'config_file' names the dialog config inside model_dir. It defaults to
        whatever this slot is configured with (TEXT_SLOTS[].config_file, itself
        defaulting to genie_config.json) — pass it when the bundle you are
        switching to names its config differently from the one the slot
        started with. It applies to this switch and stays with the slot
        afterwards, the same as if it had been configured that way.

        Serialized on the TARGET SLOT's own lock only — an in-flight request
        on a different slot is unaffected. See SlotManager.switch_model for
        the unload_first tradeoff (a reliable switch that empties the slot on
        failure, vs. keeping the old model as a fallback by holding both)."""
        body = await read_json_body(request)
        slot = manager.select_by_name(body.get("slot", ""))
        model_dir_str = body.get("model_dir", "")
        unload_first = bool(body.get("unload_first", True))
        config_file = body.get("config_file") or slot.config_file
        if not model_dir_str:
            raise InvalidRequestError("'model_dir' is required.", "model_dir")

        candidate = resolve_model_path(model_dir_str, cfg.models_base_dir)
        if not (candidate / config_file).exists():
            raise InvalidRequestError(
                f"No {config_file} found under: {candidate}. Pass "
                f"'config_file' if this bundle names its dialog config "
                f"something else.", "model_dir", status_code=404)

        def _switch():
            try:
                # Set before the load, since switch_model reads it; restored
                # on failure so a slot never advertises a config it is not
                # actually running.
                previous_config_file, slot.config_file = slot.config_file, config_file
                manager.switch_model(slot, candidate, unload_first)
            except Exception as e:
                slot.config_file = previous_config_file
                logger.error(f"[{slot.name}] Model switch to {candidate} failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
            finally:
                manager.status[slot.name] = {"phase": "idle", "detail": ""}

        await _admin_with_slot_lock(
            request, slot, _switch, cfg.warmup_join_timeout_s,
            require_loaded=False,
            busy_detail=f"Slot '{slot.name}' busy; could not acquire lock for "
                        f"model switch within {cfg.warmup_join_timeout_s}s.")

        return {"status": "switched", "slot": slot.name,
                "model": slot.active_model_id, "template": slot.chat_template}

    def _require_context_room(slot, prompt_text: str) -> None:
        """400 instead of a silent empty 200 when the prompt alone fills the
        model's context (engine.prompt_tokens_over_context)."""
        over = engine.prompt_tokens_over_context(slot, prompt_text)
        if over is None:
            return
        n, ctx = over
        raise InvalidRequestError(
            f"This model's maximum context length is {ctx} tokens. However, "
            f"your messages resulted in {n} tokens. Please reduce the length "
            "of the messages (a large `tools` block counts towards this).",
            "messages", code="context_length_exceeded")

    # ------------------------------------------------------------ completions

    def _normalize_prompts(prompt, slot) -> list[tuple[str, list | None]]:
        """OpenAI /v1/completions accepts a string, an array of strings, an
        array of token ids, or an array of token-id arrays (lm_eval's
        local-completions with tokenizer_backend=huggingface sends the token
        forms). Returns [(text, token_ids_or_None), ...]; token ids are
        decoded with the slot's own tokenizer and kept for prompt scoring
        (echo+logprobs), where exact id alignment matters."""
        def decode(ids: list) -> tuple[str, list]:
            if slot.tokenizer is None:
                raise InvalidRequestError(
                    "Token-id prompts need the model tokenizer, which is not "
                    "loaded on this server (install 'tokenizers' and ensure "
                    "genie_config.json points at tokenizer.json).", "prompt")
            return slot.tokenizer.decode(list(ids), skip_special_tokens=False), list(ids)

        if isinstance(prompt, str):
            return [(prompt, None)]
        if isinstance(prompt, list):
            if not prompt:
                return [("", None)]
            if all(isinstance(p, str) for p in prompt):
                return [(p, None) for p in prompt]
            if all(isinstance(p, int) for p in prompt):
                return [decode(prompt)]
            if all(isinstance(p, list) and all(isinstance(t, int) for t in p)
                   for p in prompt):
                return [decode(p) for p in prompt]
        raise InvalidRequestError(
            "'prompt' must be a string, an array of strings, or token id "
            "array(s)", "prompt")

    async def _score_prompt(slot, model_name: str, request_id: str,
                            text: str, ids: list | None, top_n: int,
                            extra_tokens: int = 0, epoch: int | None = None):
        """Prompt scoring (lm_eval loglikelihood): prefill only the first
        token, then teacher-force every following prompt token through the
        decode loop, recording P(token_i | tokens_<i) from the custom
        sampler's logits at each step. Exact, but runs the whole prompt at
        decode speed. Returns a finished choice dict (no `index`).

        `extra_tokens` (0 or 1) continues past the prompt and greedily
        generates that many real tokens, appending them to the logprobs
        arrays. lm_eval's loglikelihood sends `max_tokens: 1` with `echo`
        and then drops the last entry (`token_logprobs[ctxlen:-1]`), so
        without a trailing generated token every score would silently lose
        its final continuation token."""
        if ids is None:
            ids = slot.tokenizer.encode(text).ids
        if not ids:
            raise InvalidRequestError("cannot score an empty prompt", "prompt")
        if len(ids) > cfg.prompt_logprobs_max_tokens:
            raise InvalidRequestError(
                f"prompt has {len(ids)} tokens; prompt scoring is capped at "
                f"{cfg.prompt_logprobs_max_tokens} (PROMPT_LOGPROBS_MAX_TOKENS).",
                "prompt")

        n_steps = len(ids) - 1 + extra_tokens
        if n_steps == 0:
            # Nothing to score: the first token's logprob is undefined.
            lp = logprobs_mod.completions_logprobs(
                slot.tokenizer, [], first_token_id=ids[0])
            return {"text": text, "logprobs": lp, "finish_reason": "length"}

        status_entry = {"phase": "scoring prompt", "detail": f"0/{len(ids) - 1}"}
        manager.status[slot.name] = status_entry
        # Steps past the forced list fall through to greedy in the collector,
        # which is what produces the trailing generated token.
        collector = LogprobsCollector(
            top_n=top_n, forced_tokens=ids[1:], status_entry=status_entry)
        first_text = slot.tokenizer.decode([ids[0]], skip_special_tokens=False)
        score_params = GenParams(max_tokens=n_steps, stop=[])
        timeout_s = max(cfg.inference_timeout_s,
                        len(ids) * _SCORING_SECONDS_PER_TOKEN)
        gen = Generation(request_id, slot, state.lib, epoch=epoch)
        engine.start_generation(
            state.lib, slot, QueryPlan(full_prompt=first_text), score_params,
            gen, None, timeout_s, collector=collector)
        try:
            await _collect_or_raise(gen, state, timeout_s=timeout_s,
                                    collector=collector)
        finally:
            manager.status[slot.name] = {"phase": "idle", "detail": ""}
        if len(collector.results) != n_steps:
            # Every entry is paired with ids[i + 1] by position, so a short
            # list would shift every score. Fail instead of returning them.
            reason = f": {gen.error}" if gen.error else ""
            raise HTTPException(
                status_code=500,
                detail=f"prompt scoring stopped after {len(collector.results)} "
                       f"of {n_steps} steps{reason} (the runtime stopped calling "
                       f"the logits callback, or a forced prompt token ended "
                       f"generation) [{request_id}]")
        lp = logprobs_mod.completions_logprobs(
            slot.tokenizer, collector.results, first_token_id=ids[0])
        echoed = text
        if extra_tokens and len(collector.results) > len(ids) - 1:
            # OpenAI echoes prompt + generation; keep `text` consistent with
            # the tokens array.
            echoed += "".join(lp["tokens"][len(ids):])
        return {"text": echoed, "logprobs": lp, "finish_reason": "length"}

    @app.post("/completions")
    @app.post("/v1/completions")
    async def completions(request: Request):
        """Raw text completion — lm_eval's local-completions backend. The
        prompt goes to GenieDialog_query with no chat template and no prefix
        caching. Routes to a slot via 'model' (or an explicit 'slot' field);
        defaults to the primary slot."""
        body = await read_json_body(request)
        requested_model = body.get("model", KNOWN_MODEL_ID)
        slot = manager.select_for_request(body, requested_model)
        manager.require_loaded(slot)
        # Everything below is planned from the slot's current model. Read the
        # epoch first, so a swap that lands before the lock is taken is seen.
        epoch = slot.epoch
        # The response reports the model that actually answered, not the
        # string the client sent. They differ whenever a client routes with an
        # alias — lm_eval sends a fixed placeholder for every request — and
        # after a hot-swap, when the slot holds something other than what
        # env_config.json names. Echoing the request made two runs of the same
        # prompt against two different models look like one model behaving
        # nondeterministically; it cost a session to work out. OpenAI reports
        # the resolved model too (ask for "gpt-4", get "gpt-4-0613").
        model_name = slot.active_model_id
        _reject_unsupported(body, "completions")
        params = _parse_gen_params(body)
        stream = _flag(body, "stream")
        include_usage = _include_usage(body)
        echo = _flag(body, "echo")
        top_n = _completions_top_n(body)

        prompts = _normalize_prompts(body.get("prompt", ""), slot)

        if top_n is not None:
            _require_logprobs_support(slot)
            if stream:
                raise InvalidRequestError(
                    "logprobs are not supported with stream=true on this "
                    "server", "logprobs")

        # Prompt scoring (lm_eval loglikelihood shape: echo + logprobs).
        # Gated behind an explicit switch — one scoring request occupies its
        # slot for len(prompt)/decode-rate seconds (see _score_prompt).
        if top_n is not None and echo:
            if not state.prompt_logprobs_enabled:
                raise InvalidRequestError(
                    "echo+logprobs (prompt scoring, used by lm_eval "
                    "loglikelihood tasks) is disabled on this server. Enable "
                    "it with: POST /v1/server/prompt_logprobs "
                    '{"enabled": true} — note each request then runs its '
                    "whole prompt at decode speed.", "echo")
            # lm_eval's loglikelihood sends max_tokens=1 (and slices the
            # trailing entry off again), so 1 is part of the shape this mode
            # exists to serve. Anything beyond that is a generation request
            # wearing a scoring request's clothes.
            extra_tokens = body.get("max_tokens")
            if extra_tokens is None:
                extra_tokens = body.get("max_completion_tokens")
            extra_tokens = int(extra_tokens or 0)
            if extra_tokens not in (0, 1):
                raise InvalidRequestError(
                    "echo+logprobs is a prompt-scoring mode; max_tokens must "
                    "be 0, 1 (lm_eval's loglikelihood shape) or omitted.",
                    "max_tokens")
            request_id = f"cmpl-{uuid.uuid4()}"
            choices, total_pt = [], 0
            for index, (text, ids) in enumerate(prompts):
                choice = await _score_prompt(
                    slot, model_name, f"{request_id}-{index}", text, ids, top_n,
                    extra_tokens=extra_tokens, epoch=epoch)
                choice["index"] = index
                choices.append(choice)
                total_pt += len(choice["logprobs"]["tokens"]) - extra_tokens
            return protocol.completion_response(
                request_id, model_name, choices, total_pt,
                extra_tokens * len(prompts))

        _require_positive_max_tokens(params)

        if stream:
            if len(prompts) > 1:
                raise InvalidRequestError(
                    "streaming supports a single prompt per request", "prompt")
            prompt = prompts[0][0]
            _require_context_room(slot, prompt)
            req_params = replace(params, max_tokens=engine.default_max_tokens(
                slot, prompt, params.max_tokens, cfg.default_max_tokens_cap))
            request_id = f"cmpl-{uuid.uuid4()}"
            gen = Generation(request_id, slot, state.lib, epoch=epoch)
            engine.start_generation(
                state.lib, slot, QueryPlan(full_prompt=prompt), req_params, gen,
                None, cfg.inference_timeout_s)
            preamble = [protocol.completion_chunk(request_id, model_name, prompt)] \
                if echo else []
            return StreamingResponse(
                _sse_stream(
                    request, state, gen, model_name,
                    make_delta=lambda text: protocol.completion_chunk(
                        request_id, model_name, text),
                    make_final=lambda reason: protocol.completion_chunk(
                        request_id, model_name, "", finish_reason=reason),
                    preamble=preamble,
                    include_usage=include_usage,
                    prompt_tokens=slot.count_prompt_tokens(prompt),
                    usage_object_type="text_completion",
                ),
                media_type="text/event-stream")

        # Non-streaming: prompts run sequentially, one choice per prompt.
        request_id = f"cmpl-{uuid.uuid4()}"
        choices, total_pt, total_ct = [], 0, 0
        for index, (prompt, _ids) in enumerate(prompts):
            _require_context_room(slot, prompt)
            req_params = replace(params, max_tokens=engine.default_max_tokens(
                slot, prompt, params.max_tokens, cfg.default_max_tokens_cap))
            collector = None
            if top_n is not None:
                collector = LogprobsCollector(
                    top_n=top_n, temperature=params.temperature,
                    top_p=params.top_p, top_k=params.top_k, seed=params.seed)
            gen = Generation(f"{request_id}-{index}", slot, state.lib, epoch=epoch)
            engine.start_generation(
                state.lib, slot, QueryPlan(full_prompt=prompt), req_params, gen,
                None, cfg.inference_timeout_s, collector=collector)
            text = await _collect_or_raise(gen, state, collector=collector)
            total_pt += slot.count_prompt_tokens(prompt)
            total_ct += gen.completion_tokens
            choices.append({
                "text": (prompt + text) if echo else text,
                "index": index,
                "logprobs": logprobs_mod.completions_logprobs(
                    slot.tokenizer, collector.results)
                if collector is not None else None,
                "finish_reason": gen.finish_reason,
            })
        return protocol.completion_response(
            request_id, model_name, choices, total_pt, total_ct)

    # ------------------------------------------------------------ chat

    async def _vlm_chat(request: Request, body: dict, requested_model: str,
                        params: GenParams, stream: bool):
        """Chat requests whose messages contain image or video parts — routed through
        the GenieNode/GeniePipeline path. Single-turn only; the request's
        max_tokens/stop plus prefix-cache/LoRA/tools do not apply (no SDK APIs
        for them). Generation length is bounded by the slot's static
        VLM_SLOTS[].max_tokens instead; hitting it, or filling the context,
        comes back as finish_reason="length"."""
        if not vlm.VLM_AVAILABLE:
            return openai_error(
                503, "VLM support is not available on this server "
                     "(numpy/Pillow missing at startup).", "server_error")
        if _flag(body, "logprobs"):
            raise InvalidRequestError(
                "logprobs are not supported for VLM (image) requests", "logprobs")
        vslot = manager.select_vlm_for_request(body, requested_model)
        if vslot is None:
            raise InvalidRequestError(
                "This request has image or video content but no VLM_SLOTS "
                "are configured on this server.", "model")
        model_name = vslot.active_model_id   # the model that answers, see above
        try:
            system_text, parts, sources = vlm.extract_multimodal_parts(
                body["messages"])
            # All three raise ValueError for things the client can fix: a
            # malformed part, an undecodable one, or — when
            # VLM_VISION_BUDGET_GUARD is on — more visual input than the
            # context holds. Those checks belong here, not in the worker
            # thread, so they come back as a 400 rather than a 500. The guard
            # is off by default: overrunning wedges the slot, but hiding what
            # the SDK does is not this server's job (see plan_segments).
            #
            # Planning before decoding is deliberate: the step count comes
            # from the parts alone, so a request the guard refuses never pays
            # for turning its frames into bitmaps.
            segments = vlm.plan_segments(vslot, system_text, parts,
                                         vlm.extract_video_meta(body),
                                         guard=cfg.vlm_vision_budget_guard)
            images = await asyncio.to_thread(
                vlm.decode_media_sources, sources,
                max_image_pixels=cfg.vlm_max_image_pixels,
                max_total_pixels=cfg.vlm_max_total_pixels)
        except ValueError as e:
            raise InvalidRequestError(str(e), "messages")

        request_id = f"chatcmpl-{uuid.uuid4()}"
        # Counted before the stream branch so both paths report the same
        # prompt_tokens, and on the text path's basis: what is prefilled. The
        # rendered text segments through the tokenizer (chat-template markers
        # included, as a text slot counts its rendered prompt), the visual
        # input derived from the step count — the pipeline never reports that
        # one back, and leaving it out understated a 28-frame request's prompt
        # by 3584 tokens while reporting the same 20 as a 10-frame one — and
        # the BOS the text-encoder adds per segment. See vlm.count_prompt_tokens.
        prompt_tokens = vlm.count_prompt_tokens(vslot, segments)
        gen = Generation(request_id, vslot, state.lib)
        vlm.start_vlm_generation(state.lib, vslot, segments, images,
                                 params, gen)

        if stream:
            return StreamingResponse(
                _sse_stream(
                    request, state, gen, model_name,
                    make_delta=lambda text: protocol.chat_chunk(
                        request_id, model_name, {"content": text}),
                    make_final=lambda reason: protocol.chat_chunk(
                        request_id, model_name, {}, finish_reason=reason),
                    preamble=[protocol.chat_role_chunk(request_id, model_name)],
                    include_usage=_include_usage(body),
                    prompt_tokens=prompt_tokens,
                    abortable=False,
                ),
                media_type="text/event-stream")

        text = await _collect_or_raise(gen, state)
        return protocol.chat_response(
            request_id, model_name, text, gen.finish_reason,
            prompt_tokens, gen.completion_tokens or vslot.count_tokens(text))

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Chat completion — lm_eval's local-chat-completions backend, Open
        WebUI, OpenAI SDKs. Applies the loaded model's chat template,
        supports prefix caching for system prompts, streaming, and OpenAI
        `tools` (Hermes/Qwen3 function calling)."""
        body = await read_json_body(request)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise InvalidRequestError("'messages' must be a non-empty array", "messages")
        requested_model = body.get("model", KNOWN_MODEL_ID)
        stream = _flag(body, "stream")
        include_usage = _include_usage(body)   # also validates it for _vlm_chat
        _reject_unsupported(body, "chat")
        params = _parse_gen_params(body)
        _require_positive_max_tokens(params)

        # Qwen3-style reasoning toggle. Accepts both a flat top-level
        # `enable_thinking` (this server's shorthand) and vLLM/SGLang's
        # `chat_template_kwargs: {"enable_thinking": ...}` — the latter lets
        # clients already written for vLLM/SGLang-served Qwen3 work here
        # unchanged. chat_template_kwargs takes priority if both are given.
        chat_template_kwargs = body.get("chat_template_kwargs") or {}
        if not isinstance(chat_template_kwargs, dict):
            raise InvalidRequestError(
                "chat_template_kwargs must be an object", "chat_template_kwargs")
        enable_thinking = _flag(chat_template_kwargs, "enable_thinking",
                                _flag(body, "enable_thinking", True))

        # tools: OpenAI function calling. tool_choice is already validated
        # down to "auto"/"none"/absent by _reject_unsupported.
        tools = body.get("tools") or []
        if body.get("tool_choice") == "none":
            tools = []

        # VLM routing: any message with an image_url content part goes
        # through the GenieNode/GeniePipeline path.
        if vlm.is_vlm_request(messages):
            return await _vlm_chat(request, body, requested_model, params, stream)

        slot = manager.select_for_request(body, requested_model)
        manager.require_loaded(slot)
        # Everything below is planned from the slot's current model. Read the
        # epoch first, so a swap that lands before the lock is taken is seen.
        epoch = slot.epoch
        # The response reports the model that actually answered, not the
        # string the client sent. They differ whenever a client routes with an
        # alias — lm_eval sends a fixed placeholder for every request — and
        # after a hot-swap, when the slot holds something other than what
        # env_config.json names. Echoing the request made two runs of the same
        # prompt against two different models look like one model behaving
        # nondeterministically; it cost a session to work out. OpenAI reports
        # the resolved model too (ask for "gpt-4", get "gpt-4-0613").
        model_name = slot.active_model_id

        top_n = _chat_top_n(body)
        collector = None
        if top_n is not None:
            _require_logprobs_support(slot)
            if stream:
                raise InvalidRequestError(
                    "logprobs are not supported with stream=true on this "
                    "server", "logprobs")
            collector = LogprobsCollector(
                top_n=top_n, temperature=params.temperature,
                top_p=params.top_p, top_k=params.top_k, seed=params.seed)

        # Template comes from the SELECTED SLOT's actually-loaded model —
        # never from the client-supplied model_name (lm_eval sends a fixed
        # placeholder for every request).
        # ...and so does the tool dialect: the model in the slot decides how
        # calls are declared, emitted and parsed (tool_formats).
        msgs = templates.prepare_messages(messages, enable_thinking, tools,
                                          slot.tool_format)
        # Names this request actually declared, for tools.parse_tool_calls to
        # match a mangled or missing <tool_call> marker against (F25). Left
        # None when TOOL_CALL_RECOVERY is off, which restores the strict
        # tags-only parse.
        known_tool_names = {
            t["function"]["name"] for t in tools
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
            and isinstance(t["function"].get("name"), str)
        } if (tools and cfg.tool_call_recovery) else None
        prefix_prompt, query_prompt, cacheable = \
            templates.split_prompt_for_prefix_cache(msgs, slot.chat_template,
                                                    slot.tool_format,
                                                    bos=slot.sdk_bos_token is None)
        full_prompt = prefix_prompt + query_prompt if cacheable else query_prompt
        _require_context_room(slot, full_prompt)
        params.max_tokens = engine.default_max_tokens(
            slot, full_prompt, params.max_tokens, cfg.default_max_tokens_cap)

        plan = QueryPlan(
            full_prompt=full_prompt,
            query_prompt=query_prompt,
            cache_key=state.prefix_cache.key(prefix_prompt, slot.cache_namespace)
            if cacheable else None,
            cacheable=cacheable,
        )
        request_id = f"chatcmpl-{uuid.uuid4()}"
        gen = Generation(request_id, slot, state.lib, epoch=epoch)
        engine.start_generation(state.lib, slot, plan, params, gen,
                                state.prefix_cache, cfg.inference_timeout_s,
                                collector=collector)

        if stream:
            return StreamingResponse(
                _sse_stream(
                    request, state, gen, model_name,
                    make_delta=lambda text: protocol.chat_chunk(
                        request_id, model_name, {"content": text}),
                    make_final=lambda reason: protocol.chat_chunk(
                        request_id, model_name, {}, finish_reason=reason),
                    preamble=[protocol.chat_role_chunk(request_id, model_name)],
                    tool_filter=slot.tool_format.stream_filter(known_tool_names)
                    if tools else None,
                    include_usage=include_usage,
                    prompt_tokens=slot.count_prompt_tokens(full_prompt),
                ),
                media_type="text/event-stream")

        text = await _collect_or_raise(gen, state, collector=collector)
        tool_calls = None
        finish_reason = gen.finish_reason
        if tools:
            text, tool_calls = slot.tool_format.parse_tool_calls(
                text, known_tool_names)
            if tool_calls:
                finish_reason = "tool_calls"
        return protocol.chat_response(
            request_id, model_name, text, finish_reason,
            slot.count_prompt_tokens(full_prompt), gen.completion_tokens,
            tool_calls=tool_calls,
            logprobs=logprobs_mod.chat_logprobs(slot.tokenizer, collector.results)
            if collector is not None else None)

    # ------------------------------------------------------------ server status

    @app.get("/v1/server/idle")
    async def wait_for_server_idle(request: Request):
        """Blocks until a slot's lock is free. Use between benchmark phases.
        ?slot=<name> selects the slot (default: primary)."""
        slot_name = request.query_params.get("slot", "")
        slot = manager.select_by_name(slot_name)

        def _try():
            ok = slot.lock.acquire(timeout=cfg.inference_timeout_s)
            if ok:
                slot.lock.release()
            return ok

        if await asyncio.to_thread(_try):
            return {"status": "idle", "slot": slot.name}
        raise HTTPException(
            status_code=503,
            detail=f"Slot '{slot.name}' did not become idle within "
                   f"{cfg.inference_timeout_s}s")

    @app.get("/v1/server/status")
    async def get_server_status():
        """Current per-slot processing phase (non-blocking). Top-level
        'phase'/'detail' mirror the PRIMARY slot for bench_ttft.py
        compatibility; 'slots' is the full per-slot breakdown. Each slot's
        context_occupancy is only sampled when its lock is immediately free,
        so this endpoint never stalls behind an in-flight generation."""
        slot_reports = []
        for s in manager.slots:
            occ = None
            if s.handle is not None and s.lock.acquire(timeout=0):
                try:
                    occ = state.lib.get_context_occupancy(s.handle)
                finally:
                    s.lock.release()
            st = manager.status.get(s.name, {"phase": "idle", "detail": ""})
            slot_reports.append({
                "name": s.name,
                "device_id": s.device_id,
                "active_model": s.active_model_id,
                "active_lora": s.active_lora_adapter,
                "loaded": s.handle is not None,
                "phase": st.get("phase", "idle"),
                "detail": st.get("detail", ""),
                "context_occupancy": occ,
            })

        # A VLM-only deployment has no text slots, so there is no primary to
        # mirror — report an idle shape rather than raising IndexError.
        primary = slot_reports[0] if slot_reports else {
            "phase": "idle", "detail": "", "active_model": None,
            "active_lora": None, "context_occupancy": None,
        }
        return {
            "phase": primary["phase"],
            "detail": primary["detail"],
            "active_model": primary["active_model"],
            "active_lora": primary["active_lora"],
            "context_occupancy": primary["context_occupancy"],
            "slots": slot_reports,
            "vlm_slots": [
                {"name": v.name, "device_id": v.device_id,
                 "active_model": v.active_model_id, "spec": v.spec.name,
                 "layout": v.layout.source}
                for v in manager.vlm_slots
            ],
        }

    # ------------------------------------------------------------ prefix cache

    @app.get("/v1/prefix/cache")
    async def list_prefix_cache():
        return {"entries": state.prefix_cache.list_entries()}

    @app.delete("/v1/prefix/cache/{key}")
    async def delete_prefix_cache(key: str):
        if state.prefix_cache.delete(key):
            return {"deleted": key}
        raise HTTPException(status_code=404, detail=f"Cache key not found: {key}")

    @app.post("/v1/prefix/warmup")
    async def warmup_prefix_cache(request: Request):
        """Pre-populates the prefix KV cache for a given system prompt, on
        the slot selected via 'model'. Returns 200 when done, 202 if still
        running (poll /v1/prefix/cache)."""
        body = await read_json_body(request)
        slot = manager.select(body.get("model", KNOWN_MODEL_ID))
        manager.require_loaded(slot)
        epoch = slot.epoch
        system_prompt = body.get("system_prompt", "")
        if not system_prompt:
            raise InvalidRequestError("'system_prompt' is required.", "system_prompt")

        # A chat request with enable_thinking=false gets "/no_think" appended
        # to its system turn (templates.prepare_messages), which changes the
        # cached prefix and therefore its key. Warming the raw prompt and then
        # sending enable_thinking=false is a silent, permanent MISS — so take
        # the same flag here and warm the variant the caller will actually use.
        enable_thinking = _flag(body, "enable_thinking", True)
        messages = templates.prepare_messages(
            [{"role": "system", "content": system_prompt}],
            enable_thinking=enable_thinking)
        prefix_prompt, _, cacheable = templates.split_prompt_for_prefix_cache(
            messages, slot.chat_template, bos=slot.sdk_bos_token is None)
        if not cacheable or not prefix_prompt:
            # Name the actual reason for this template — quoting Llama2's
            # would just misdirect someone debugging a Gemma slot.
            why = {
                "llama2": "Llama2/Mistral embeds the system prompt in [INST]",
                "gemma": "Gemma 2/3 prepends the system prompt to the first user turn",
            }.get(slot.chat_template,
                  "this template has no separable system prefix")
            raise InvalidRequestError(
                f"Slot '{slot.name}' model '{slot.active_model_id}' "
                f"(template={slot.chat_template}) does not support prefix "
                f"caching ({why}).",
                status_code=422)

        cache_key = state.prefix_cache.key(prefix_prompt, slot.cache_namespace)
        if state.prefix_cache.exists(cache_key):
            return {"status": "already_cached", "key": cache_key, "slot": slot.name,
                    "enable_thinking": enable_thinking}

        error_msg = ""

        def do_warmup():
            nonlocal error_msg
            try:
                with slot.lock:
                    if slot.handle is None or slot.epoch != epoch:
                        # The key names the model and adapter the prefix was
                        # rendered for; saving this dialog's KV under it now
                        # would poison the cache for that namespace.
                        raise RuntimeError(
                            f"slot '{slot.name}' changed (model switch or "
                            "LoRA update) while the warmup waited; resend it")
                    manager.status[slot.name] = {"phase": "resetting dialog", "detail": ""}
                    state.lib.reset(slot.handle)
                    engine.warm_up_prefix(state.lib, slot, prefix_prompt,
                                          cache_key, state.prefix_cache,
                                          manager.status)
            except Exception as e:
                error_msg = str(e)
            finally:
                manager.status[slot.name] = {"phase": "idle", "detail": ""}

        t = threading.Thread(target=do_warmup, daemon=True)
        t.start()
        await asyncio.to_thread(t.join, cfg.warmup_join_timeout_s)

        if state.prefix_cache.exists(cache_key):
            return {"status": "cached", "key": cache_key, "slot": slot.name,
                    "enable_thinking": enable_thinking}
        if t.is_alive():
            logger.warning(f"[{slot.name}] Warmup still running after "
                           f"{cfg.warmup_join_timeout_s}s  key={cache_key}")
            return JSONResponse(
                status_code=202,
                content={"status": "warming", "key": cache_key, "slot": slot.name,
                         "message": "Warmup still in progress. Poll /v1/prefix/cache."})
        raise HTTPException(status_code=500, detail=f"Prefix warmup failed. {error_msg}")

    # ------------------------------------------------------------ prompt logprobs

    @app.get("/v1/server/profile")
    async def get_profile(request: Request):
        """The SDK's own KPIs for the most recent query on a slot.

        Deliberately a separate endpoint rather than a field on the chat or
        completion response: those shapes are the OpenAI contract, and clients
        validate them. Everything non-OpenAI this server offers lives under
        /v1/server/, so a profiling client asks for it explicitly and an
        OpenAI client never sees it.

        `?slot=<name>` picks the slot (defaults to the primary one). Requires
        GENIE_PROFILE=true in env_config.json — the profiler binds to the
        dialog config, so it cannot be switched on at runtime."""
        slot = manager.select_by_name(request.query_params.get("slot", ""))
        if slot.profile is None:
            raise InvalidRequestError(
                "profiling is disabled; set GENIE_PROFILE=true in "
                "env_config.json and restart (the profiler binds to the "
                "dialog at creation time).", "slot", status_code=409)
        try:
            raw = json.loads(state.lib.get_profile_json(slot.profile))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        summary = protocol.profile_summary(raw)
        # The SDK does not profile GenieDialog_save/restore at all (no such
        # event type; `applyEngineState` is a different path), so the prefix
        # cache's cost is timed by the server and reported separately —
        # never mixed into the SDK's own numbers.
        host = {}
        if state.prefix_cache.last_restore_ms is not None:
            host["restore_state_ms"] = round(state.prefix_cache.last_restore_ms, 3)
        if state.prefix_cache.last_save_ms is not None:
            host["save_state_ms"] = round(state.prefix_cache.last_save_ms, 3)
        return {"slot": slot.name, "model": slot.active_model_id,
                "summary": summary, "host_measured": host, "profile": raw}

    @app.get("/v1/server/prompt_logprobs")
    async def get_prompt_logprobs():
        """Whether prompt scoring (echo+logprobs teacher forcing, used by
        lm_eval loglikelihood tasks) is currently enabled."""
        return {"enabled": state.prompt_logprobs_enabled,
                "max_tokens": cfg.prompt_logprobs_max_tokens}

    @app.post("/v1/server/prompt_logprobs")
    async def set_prompt_logprobs(request: Request):
        """Body: {"enabled": true|false}. Enable before an lm_eval
        loglikelihood run, disable after — each scoring request occupies its
        slot for len(prompt)/decode-rate seconds, so this stays off unless a
        measurement is actually running (also settable at startup via
        env_config.json's PROMPT_LOGPROBS)."""
        body = await read_json_body(request)
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            raise InvalidRequestError("'enabled' (boolean) is required.", "enabled")
        state.prompt_logprobs_enabled = enabled
        logger.info(f"Prompt logprobs (teacher-forcing scoring) "
                    f"{'ENABLED' if enabled else 'disabled'}")
        return {"enabled": enabled}

    # ------------------------------------------------------------ LoRA

    def _lora_names(body: dict, *keys: str) -> tuple:
        """'engine' (default "primary") and the required string fields named
        in keys, checked before anything reaches the SDK: a non-string there
        used to be a 500 (str.encode on a list, or an unhashable dict key)."""
        engine_role = body.get("engine", "primary")
        if not isinstance(engine_role, str) or not engine_role:
            raise InvalidRequestError("'engine' must be a non-empty string.",
                                      "engine")
        values = []
        for key in keys:
            value = body.get(key, "")
            if not value:
                raise InvalidRequestError(f"'{key}' is required.", key)
            if not isinstance(value, str):
                raise InvalidRequestError(f"'{key}' must be a string.", key)
            values.append(value)
        return (engine_role, *values)

    @app.post("/v1/lora/apply")
    async def apply_lora(request: Request):
        """Body: {"slot": "...", "model": "...", "engine": "primary",
        "lora_adapter_name": "..."}

        'engine' is the engine role the adapter targets (typically "primary"
        for a single-engine dialog — check your genie_config.json).

        Slot selection follows the same rule as the chat endpoints: an
        explicit 'slot' wins, otherwise 'model' routes by loaded model name
        and an unknown one falls back to the primary slot. Two slots holding
        the same model directory are indistinguishable by 'model', so 'slot'
        is the only way to address the second one."""
        body = await read_json_body(request)
        slot = manager.select_for_request(body, body.get("model", ""))
        manager.require_loaded(slot)
        engine_role, adapter = _lora_names(body, "lora_adapter_name")

        def _apply():
            ret = state.lib.apply_lora(slot.handle, engine_role, adapter)
            if ret != STATUS_SUCCESS:
                raise HTTPException(status_code=500,
                                    detail=f"GenieDialog_applyLora failed: {ret}")
            # Read back from the SDK rather than trust the request blindly.
            slot.active_lora_adapter = state.lib.get_applied_lora(slot.handle)
            # lora_strengths is NOT cleared here: measured on the board, applying
            # another adapter keeps the alphas set earlier (alpha0/alpha1 belong
            # to the dialog, and the two adapters share the names). Only a
            # release (below) puts them back.
            slot.epoch += 1  # the prefix-cache namespace just changed

        await _admin_with_slot_lock(request, slot, _apply, cfg.inference_timeout_s)
        return {"status": "applied", "slot": slot.name, "engine": engine_role,
                "lora_adapter_name": slot.active_lora_adapter}

    @app.post("/v1/lora/strength")
    async def set_lora_strength(request: Request):
        """Body: {"model": "...", "engine": "primary", "tensor_name": "...", "alpha": 1.0}"""
        body = await read_json_body(request)
        slot = manager.select_for_request(body, body.get("model", ""))
        manager.require_loaded(slot)
        alpha = body.get("alpha")
        if not body.get("tensor_name") or alpha is None:
            raise InvalidRequestError("'tensor_name' and 'alpha' are required.")
        engine_role, tensor_name = _lora_names(body, "tensor_name")
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha)):
            raise InvalidRequestError(
                f"'alpha' must be a finite number, got {alpha!r}.", "alpha")

        def _set():
            # The SDK answers success for a name the model does not have (on
            # an engine with a scheduler it is taken as a CB adapter-order name
            # and changes nothing), so the name is checked against the config.
            # Checked here, under the lock, against the model the call will
            # actually reach: a switch that won the lock first would otherwise
            # have the old model's names vouch for the new one. A name set
            # before its adapter is applied is fine: the SDK keeps it and
            # writes it when the adapter is applied.
            known = lora_alpha_names(slot.dialog_cfg, engine_role)
            if known is not None and tensor_name not in known:
                raise InvalidRequestError(
                    f"'{tensor_name}' is not a LoRA alpha of engine "
                    f"'{engine_role}' on slot '{slot.name}'. "
                    + (f"Its config declares: {sorted(known)}." if known else
                       "Its config declares no LoRA adapters."),
                    "tensor_name")
            ret = state.lib.set_lora_strength(slot.handle, engine_role,
                                              tensor_name, float(alpha))
            if ret != STATUS_SUCCESS:
                raise HTTPException(status_code=500,
                                    detail=f"GenieDialog_setLoraStrength failed: {ret}")
            # A prefix KV computed at the old strength is wrong at the new one,
            # so the strength is part of the cache namespace, and a request
            # planned against the old one must not run (see Slot.epoch).
            # Keyed by the role the SDK acts on: "target" and "primary" are one
            # engine, and two keys for one alpha let two different strengths
            # produce the same namespace.
            key = f"{canonical_engine_role(engine_role)}/{tensor_name}"
            slot.lora_strengths[key] = float(alpha)
            slot.epoch += 1

        await _admin_with_slot_lock(request, slot, _set, cfg.inference_timeout_s)
        return {"status": "applied", "slot": slot.name, "engine": engine_role,
                "tensor_name": tensor_name, "alpha": alpha}

    @app.post("/v1/lora/release")
    async def release_lora(request: Request):
        """Frees the adapter's memory; re-applying later reloads from disk."""
        body = await read_json_body(request)
        slot = manager.select_for_request(body, body.get("model", ""))
        manager.require_loaded(slot)
        engine_role, adapter = _lora_names(body, "lora_adapter_name")

        def _release():
            ret = state.lib.release_lora_memory(slot.handle, engine_role, adapter)
            if ret != STATUS_SUCCESS:
                raise HTTPException(
                    status_code=500,
                    detail=f"GenieDialog_releaseLoraMemory failed: {ret}")
            slot.active_lora_adapter = state.lib.get_applied_lora(slot.handle)
            slot.lora_strengths = {}
            slot.epoch += 1

        await _admin_with_slot_lock(request, slot, _release, cfg.inference_timeout_s)
        return {"status": "released", "slot": slot.name, "engine": engine_role,
                "lora_adapter_name": adapter}

    @app.get("/v1/lora/current")
    async def get_current_lora(request: Request):
        """Currently-applied LoRA adapter name, read live from the SDK (""
        == base model). ?slot= selects the slot, ?model= falls back to
        model-name routing."""
        qp = request.query_params
        slot = manager.select_for_request({"slot": qp.get("slot", "")},
                                          qp.get("model", ""))
        manager.require_loaded(slot)
        try:
            name = await run_with_slot_lock(
                slot, lambda: state.lib.get_applied_lora(slot.handle), timeout=1.0)
        except SlotBusyError:
            # Don't stall a status check behind a long generation.
            return {"slot": slot.name,
                    "lora_adapter_name": slot.active_lora_adapter, "live": False}
        return {"slot": slot.name, "lora_adapter_name": name, "live": True}

    # ------------------------------------------------------------ performance

    @app.get("/v1/server/performance_policy")
    async def get_performance_policy(request: Request):
        """?model= selects the slot."""
        slot = manager.select(request.query_params.get("model", ""))
        manager.require_loaded(slot)
        try:
            val = await run_with_slot_lock(
                slot, lambda: state.lib.get_performance_policy(slot.handle),
                timeout=1.0)
        except SlotBusyError:
            return {"slot": slot.name, "policy": None, "live": False}
        return {"slot": slot.name,
                "policy": PERFORMANCE_POLICY_NAMES.get(val, val),
                "raw_value": val, "live": True}

    @app.post("/v1/server/performance_policy")
    async def set_performance_policy(request: Request):
        """Body: {"model": "...", "policy": "burst"} — pin to "burst" before a
        timed benchmark run for reproducible numbers."""
        body = await read_json_body(request)
        slot = manager.select(body.get("model", ""))
        manager.require_loaded(slot)
        policy = body.get("policy", "")
        if policy not in PERFORMANCE_POLICIES:
            raise InvalidRequestError(
                f"'policy' must be one of: {sorted(PERFORMANCE_POLICIES)}", "policy")
        def _set():
            ret = state.lib.set_performance_policy(slot.handle,
                                                   PERFORMANCE_POLICIES[policy])
            if ret != STATUS_SUCCESS:
                raise HTTPException(
                    status_code=500,
                    detail=f"GenieDialog_setPerformancePolicy failed: {ret}")

        await _admin_with_slot_lock(request, slot, _set, cfg.inference_timeout_s)
        return {"status": "applied", "slot": slot.name, "policy": policy}

    return app

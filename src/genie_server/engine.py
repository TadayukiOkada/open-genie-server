"""The generation engine: one code path for streaming and non-streaming.

A Generation object bridges the SDK's synchronous token callback (invoked
from inside the blocking GenieDialog_query C call, on a worker thread) into
an asyncio.Queue that both the SSE generator and the sync collector consume.

Responsibilities handled here, in order, all under the target slot's lock:
lock acquisition (abort-aware), watchdog, dialog reset, per-request SDK
parameters (max tokens / stop sequences / sampling), prefix-cache routing,
query, and finish_reason determination.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from . import capi
from .capi import GenieLib, make_sampler_params
from .prefix_cache import PrefixCache
from .slots import Slot

logger = logging.getLogger(__name__)


@dataclass
class GenParams:
    """Per-request generation parameters (OpenAI names/semantics)."""
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None


@dataclass
class QueryPlan:
    """What to feed GenieDialog_query, including the prefix-cache split."""
    full_prompt: str
    query_prompt: str = ""   # remainder after the cached prefix
    cache_key: str | None = None
    cacheable: bool = False


def ensure_logprobs_sampler(lib: GenieLib, slot: Slot) -> None:
    """Registers this slot's custom sampler callback (once per process).
    The registered trampoline dispatches to whatever LogprobsCollector is
    currently active on the slot — collectors are per-request, set under
    slot.lock by _locked_query."""
    if slot.logprobs_registered:
        return

    def on_logits(logits_addr, n_floats, num_tokens):
        collector = slot.active_collector
        if collector is None:
            logger.error(f"[{slot.name}] custom sampler fired with no collector")
            return [0]
        return collector.on_logits(logits_addr, n_floats, num_tokens)

    lib.register_custom_sampler(slot.sampler_callback_name, on_logits)
    slot.logprobs_registered = True


class Generation:
    """One in-flight generation and its thread/asyncio bridge."""

    def __init__(self, request_id: str, slot, lib: GenieLib):
        self.request_id = request_id
        self.slot = slot
        self._lib = lib
        self._loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = threading.Event()
        self.aborted = threading.Event()
        # Set only while OUR GenieDialog_query is running on this slot.
        # abort() checks it so that a client disconnecting while still WAITING
        # for the slot lock can never signal-abort someone else's generation
        # in flight on the same slot.
        self.query_active = threading.Event()
        self.finish_reason = "stop"
        self.error: str | None = None
        # Set by the watchdog when it fires, whatever the query was doing at
        # that moment. The SDK reports a watchdog abort exactly as it reports
        # a client's, so this flag is what tells them apart -- but on its own
        # it is not the verdict: the timer can also fire during setup, or
        # just after the query has already finished normally.
        self.timed_out = False
        # The verdict, decided once by the engine (_locked_query) from
        # timed_out plus what the query actually returned. True means the
        # output is incomplete because of the timeout; error then says so.
        # Both the sync and the streaming path read this, never timed_out.
        self.timeout_error = False
        # A SENTENCE_ABORT reached the token callback: the SDK itself says
        # the query was cut short, whatever status it returns.
        self.abort_seen = False
        # Set when a logprobs request emitted a token without the custom
        # sampler's logits callback having run (see _locked_query).
        self.logprobs_unsupported = False
        self.completion_tokens = 0
        self.cache_state = "NONE"
        self.query_started_at: float | None = None
        self.ttft_logged = False

    def put_threadsafe(self, item) -> None:
        try:
            if not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            pass

    def abort(self) -> None:
        """Client went away: stop waiting for the lock, and abort the query
        only if it is actually ours and running."""
        self.aborted.set()
        if self.query_active.is_set() and getattr(self.slot, "handle", None) is not None:
            ret = self._lib.signal_abort(self.slot.handle)
            if ret != capi.STATUS_SUCCESS:
                logger.warning(f"GenieDialog_signal failed [{self.request_id}]: {ret}")

    def abort_on_timeout(self) -> None:
        """Watchdog: the query ran past inference_timeout_s."""
        self.timed_out = True
        self.abort()

    def on_token(self, token: str, code: int) -> None:
        """SDK token callback (worker thread, inside GenieDialog_query)."""
        try:
            if code == capi.SENTENCE_ABORT:
                self.abort_seen = True
            if token and code != capi.SENTENCE_ABORT:
                if not self.ttft_logged and self.query_started_at is not None:
                    ms = (time.perf_counter() - self.query_started_at) * 1000
                    logger.info(f"TTFT [{self.request_id}] slot={self.slot.name} "
                                f"cache={self.cache_state:4s} {ms:.1f}ms")
                    self.ttft_logged = True
                self.completion_tokens += 1
                self.put_threadsafe(token)
        except Exception as e:
            logger.error(f"Exception in token callback [{self.request_id}]: {e}")

    async def collect_text(self, timeout_s: float) -> str:
        """Sync path: drains the queue into one string. Raises TimeoutError
        on overall timeout, RuntimeError if the query failed outright."""
        chunks: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Inference timed out [{self.request_id}]")
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Inference timed out [{self.request_id}]") from None
            if item is None:
                break
            chunks.append(item)
        if self.timeout_error:
            # Whatever arrived is a truncated output, not a completion.
            raise TimeoutError(f"Inference timed out [{self.request_id}]")
        if self.error and not chunks:
            raise RuntimeError(self.error)
        return "".join(chunks)


def start_generation(
    lib: GenieLib,
    slot: Slot,
    plan: QueryPlan,
    params: GenParams,
    generation: Generation,
    prefix_cache: PrefixCache | None,
    inference_timeout_s: float,
    collector=None,
) -> None:
    """Kicks off one text generation on a worker thread. Tokens (and a final
    None sentinel) arrive on generation.queue; generation.finish_reason /
    .error are final once the sentinel is queued.

    `collector` (a logprobs.LogprobsCollector) switches the slot's sampler
    to the custom (logits-observing) callback for this request only; the
    basic sampler with the model's defaults is restored before the lock is
    released."""

    def worker() -> None:
        try:
            # Abort-aware lock wait: give up promptly if the client is gone.
            acquired = False
            while not generation.aborted.is_set():
                if slot.lock.acquire(timeout=0.2):
                    acquired = True
                    break
            if not acquired:
                logger.info(f"Client disconnected before lock [{generation.request_id}]")
                return
            try:
                _locked_query(lib, slot, plan, params, generation,
                              prefix_cache, inference_timeout_s, collector)
            finally:
                slot.lock.release()
        except Exception as e:
            logger.error(f"Exception in inference worker [{generation.request_id}]: {e}")
            generation.error = str(e)
        finally:
            generation.put_threadsafe(None)
            generation.done.set()

    threading.Thread(target=worker, daemon=True).start()


def _locked_query(lib: GenieLib, slot: Slot, plan: QueryPlan, params: GenParams,
                  generation: Generation, prefix_cache: PrefixCache | None,
                  inference_timeout_s: float, collector=None) -> None:
    """The actual SDK sequence, with slot.lock held."""
    watchdog = threading.Timer(inference_timeout_s, generation.abort_on_timeout)
    watchdog.start()
    try:
        lib.reset(slot.handle)
        lib.set_max_tokens(slot.handle, params.max_tokens)
        lib.set_stop_sequences(slot.handle, params.stop or None)
        if collector is not None:
            # Logprobs request: sampling moves into the collector (custom
            # sampler). temperature/top_p/top_k/seed are honored by the
            # collector itself, not the SDK sampler.
            ensure_logprobs_sampler(lib, slot)
            slot.active_collector = collector
            lib.apply_sampler_params(slot.handle, {
                "type": "custom",
                "callback-name": slot.sampler_callback_name,
            })
        else:
            lib.apply_sampler_params(slot.handle, make_sampler_params(
                slot.sampler_defaults, params.temperature, params.top_p,
                params.top_k, params.seed))

        # Prefix cache routing
        actual, sentence_code = plan.full_prompt, capi.SENTENCE_COMPLETE
        if plan.cacheable and plan.cache_key and prefix_cache is not None:
            if prefix_cache.exists(plan.cache_key) and \
                    prefix_cache.restore(lib, slot.handle, plan.cache_key):
                actual, sentence_code = plan.query_prompt, capi.SENTENCE_END
                generation.cache_state = "HIT"
            else:
                logger.info(f"Prefix MISS key={plan.cache_key[:8]} "
                            f"[{generation.request_id}]")
                generation.cache_state = "MISS"

        if generation.aborted.is_set():
            # Either the client left while we were setting up, or the
            # watchdog fired before the query started (a slow reset or
            # prefix restore). Only the second is an error anyone reads.
            if generation.timed_out:
                _fail_on_timeout(generation, inference_timeout_s,
                                 "before the query started")
            return

        on_token = generation.on_token
        if collector is not None:
            on_token = _logits_checked_on_token(lib, slot, generation, collector)

        generation.query_started_at = time.perf_counter()
        generation.query_active.set()
        try:
            ret = lib.query(slot.handle, actual, sentence_code, on_token)
        finally:
            generation.query_active.clear()
        # Taken the moment the query returns: a timer that fires after this
        # found the query already finished, and must not turn a complete
        # output into a timeout.
        timed_out_during_query = generation.timed_out

        # finish_reason is determined here, BEFORE the sentinel is queued, so
        # consumers can never observe a stale value.
        if ret < capi.STATUS_SUCCESS:
            logger.error(f"GenieDialog_query failed [{generation.request_id}]: {ret}")
            generation.error = f"GenieDialog_query failed with status {ret}"
        elif timed_out_during_query and (ret == capi.WARNING_ABORTED
                                         or generation.abort_seen):
            # Our watchdog cut the query short. Report it as a failure: a
            # partial output labeled finish_reason "stop" would be scored as
            # a finished sample. The abort is recognised by either signal the
            # SDK gives -- the WARNING_ABORTED status or a SENTENCE_ABORT
            # callback -- since which one a given runtime uses is not
            # something this server has measured on every version. A timer
            # that fired but lost the race to a normal finish (SUCCESS, no
            # abort callback) is not a timeout.
            _fail_on_timeout(generation, inference_timeout_s, "and was aborted")
        else:
            if ret > capi.STATUS_SUCCESS:
                logger.warning(f"GenieDialog_query warning [{generation.request_id}]: {ret}")
            # Only WARNING_CONTEXT_EXCEEDED means "length" — ABORTED/
            # BOUND_HANDLE/PAUSED must not be mislabeled as a token limit.
            generation.finish_reason = capi.status_to_finish_reason(ret, "stop")
            # The SDK reports hitting setMaxNumTokens as a plain successful
            # stop; detect it ourselves for OpenAI's finish_reason="length".
            if (generation.finish_reason == "stop" and params.max_tokens
                    and generation.completion_tokens >= params.max_tokens):
                generation.finish_reason = "length"
    finally:
        if collector is not None:
            # Restore the basic sampler with model defaults before anyone
            # else can touch this dialog (still under slot.lock).
            slot.active_collector = None
            lib.apply_sampler_params(
                slot.handle, make_sampler_params(slot.sampler_defaults))
        watchdog.cancel()
        watchdog.join()


def _fail_on_timeout(generation: Generation, inference_timeout_s: float,
                     when: str) -> None:
    """Records the engine's verdict that this generation timed out."""
    logger.error(f"Inference exceeded {inference_timeout_s:g}s {when} "
                 f"[{generation.request_id}]")
    generation.timeout_error = True
    generation.error = (f"Inference timed out after {inference_timeout_s:g}s "
                        f"{when}; the output is incomplete")


def _logits_checked_on_token(lib: GenieLib, slot: Slot, generation: Generation,
                             collector):
    """Token callback for a logprobs request that stops the query as soon as
    a token arrives without the logits callback having run.

    Some QAIRT runtimes (2.46 on the IQ-9075 Ubuntu packages) accept the
    custom sampler config but never call its hook, and the SDK samples on its
    own. Waiting for the whole generation to learn that would hold the slot
    for up to max_tokens steps, so the first token decides. The result is
    remembered on the slot, and later requests fail before they queue."""

    def on_token(token: str, code: int) -> None:
        if generation.logprobs_unsupported:
            return
        if token and code != capi.SENTENCE_ABORT and collector.step == 0:
            logger.error(f"[{slot.name}] logits callback was not invoked before "
                         f"the first token; logprobs are unsupported by this "
                         f"runtime [{generation.request_id}]")
            generation.logprobs_unsupported = True
            slot.logits_callback_unsupported = True
            # GenieDialog_signal(ABORT) only sets a flag, so it is safe to
            # call from inside the query's own callback.
            ret = lib.signal_abort(slot.handle)
            if ret != capi.STATUS_SUCCESS:
                logger.warning(f"GenieDialog_signal failed "
                               f"[{generation.request_id}]: {ret}")
            return
        generation.on_token(token, code)

    return on_token


def warm_up_prefix(lib: GenieLib, slot: Slot, prefix_prompt: str, cache_key: str,
                   prefix_cache: PrefixCache, status: dict) -> bool:
    """Feeds a system prefix with SENTENCE_BEGIN (prefill without
    generating), then saves the populated KV cache. Must be called with
    slot.lock held. Updates status[slot.name] for /v1/server/status polling."""
    token_count = 0

    def on_token(token: str, code: int) -> None:
        nonlocal token_count
        if token:
            token_count += 1
            status[slot.name]["detail"] = f"{token_count} tokens"

    status[slot.name] = {"phase": "prefilling prefix", "detail": "0 tokens"}
    ret = lib.query(slot.handle, prefix_prompt, capi.SENTENCE_BEGIN, on_token)
    if ret < capi.STATUS_SUCCESS:
        logger.error(f"[{slot.name}] Warmup query failed: {ret}")
        return False
    status[slot.name] = {"phase": "saving KV cache", "detail": "writing to disk…"}
    return prefix_cache.save(lib, slot.handle, cache_key)


def prompt_tokens_over_context(slot: Slot, prompt_text: str) -> tuple[int, int] | None:
    """(prompt_tokens, context_size) when the prompt cannot fit, else None.

    Without this the request still runs: default_max_tokens clamps the
    remaining window to 1, the SDK generates nothing and reports the plain
    "length" stop, and the caller gets HTTP 200 with an empty string. Open
    WebUI 0.11 walks into exactly that — its 34 built-in tool definitions
    render to ~5,800 tokens against a 4,096-token context — and shows a blank
    reply with no clue why. OpenAI answers this case with a 400
    context_length_exceeded, so we do too."""
    context_size = slot.context_size
    if not context_size:
        return None
    n = slot.count_prompt_tokens(prompt_text)
    return (n, context_size) if n >= context_size else None


def default_max_tokens(slot: Slot, prompt_text: str, max_tokens: int | None,
                       extra_cap: int) -> int | None:
    """When the client doesn't specify max_tokens, bound generation by the
    model's remaining context window (context size - prompt tokens) instead
    of leaving it fully unbounded — matching Qualcomm's qai-appbuilder
    reference server (ApplyOutputSizeBudget). An explicit client value is
    always respected. extra_cap (env_config.json's DEFAULT_MAX_TOKENS) adds
    an optional additional ceiling for deployments whose model is prone to
    runaway generation."""
    if max_tokens is not None:
        return max_tokens
    context_size = slot.context_size
    if context_size:
        remaining = max(context_size - slot.count_prompt_tokens(prompt_text), 1)
        return min(remaining, extra_cap) if extra_cap > 0 else remaining
    return extra_cap if extra_cap > 0 else None

"""Client-disconnect handling.

The defect these guard (fixed before the package split, never regression
tested): the disconnect handler used to call `GenieDialog_signal(ABORT)` on
the slot unconditionally, so a client that gave up **while still queued for
the slot lock** aborted whatever generation was in flight on that slot —
someone else's request. `Generation.query_active` is what makes an abort
apply only to our own running query.

Everything here drives the real `engine`/`app` code against the fake SDK; the
one thing it cannot cover is starlette actually dropping the socket, so
`_sse_stream` is driven with a request object that reports the disconnect.
"""

import asyncio
import threading

import pytest

from genie_server import capi
from genie_server.engine import GenParams, Generation, QueryPlan, start_generation
from fake_genie import FakeGenieLib


class BlockingGenieLib(FakeGenieLib):
    """query() emits one token and then blocks until aborted or released —
    long enough to hold the slot while a second request queues behind it."""

    def __init__(self):
        super().__init__()
        self.released = threading.Event()
        self.in_query = threading.Event()
        self.queries = 0

    def query(self, handle, text, sentence_code, on_token) -> int:
        self.queries += 1
        self.in_query.set()
        on_token("first ", capi.SENTENCE_CONTINUE)
        self.released.wait(timeout=10)
        on_token("", capi.SENTENCE_END)
        return 0

    def signal_abort(self, handle) -> int:
        rc = super().signal_abort(handle)
        self.released.set()      # a real abort makes query return promptly
        return rc


def make_slot(lib, name="cdsp0"):
    from pathlib import Path
    from genie_server.slots import Slot
    from fake_genie import FakeTokenizer
    slot = Slot(name=name, device_id=None, model_root=Path("/models/fake"))
    slot.handle = lib.create_dialog(b"{}")
    slot.dialog_cfg = {"context": {"size": 4096}}
    slot.chat_template = "chatml"
    slot.tokenizer = FakeTokenizer()
    lib.tokenizer = slot.tokenizer
    return slot


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------- Generation.abort

def test_abort_while_queued_does_not_touch_the_sdk():
    """query_active is clear: we do not own the dialog, so nothing may be
    signalled — this is the F10 guard in one assertion."""
    async def body():
        lib = FakeGenieLib()
        slot = make_slot(lib)
        gen = Generation("req-queued", slot, lib)

        gen.abort()

        assert gen.aborted.is_set()
        assert lib.abort_signals == 0
    run(body())


def test_abort_during_our_own_query_signals_once():
    async def body():
        lib = FakeGenieLib()
        slot = make_slot(lib)
        gen = Generation("req-running", slot, lib)
        gen.query_active.set()

        gen.abort()

        assert gen.aborted.is_set()
        assert lib.abort_signals == 1
    run(body())


def test_abort_with_no_handle_is_a_no_op():
    """A slot whose model was freed mid-flight must not be signalled."""
    async def body():
        lib = FakeGenieLib()
        slot = make_slot(lib)
        gen = Generation("req-unloaded", slot, lib)
        gen.query_active.set()
        slot.handle = None

        gen.abort()

        assert lib.abort_signals == 0
    run(body())


# ------------------------------------------------- two requests, one slot

def test_disconnect_while_queued_never_aborts_the_request_in_flight():
    """The regression itself: A holds the slot and is generating, B is still
    waiting for the lock when its client disappears. B must go away quietly."""
    async def body():
        lib = BlockingGenieLib()
        slot = make_slot(lib)

        gen_a = Generation("req-A", slot, lib)
        start_generation(lib, slot, QueryPlan(full_prompt="hello"),
                         GenParams(max_tokens=8), gen_a, None, 30.0)
        assert lib.in_query.wait(timeout=5), "A never reached the SDK query"

        gen_b = Generation("req-B", slot, lib)
        start_generation(lib, slot, QueryPlan(full_prompt="hello"),
                         GenParams(max_tokens=8), gen_b, None, 30.0)

        gen_b.abort()                      # B's client hung up while queued

        assert gen_b.done.wait(timeout=5), "B's worker did not give up on the lock"
        assert lib.abort_signals == 0, "B aborted A's in-flight generation"
        assert lib.queries == 1, "B ran a query despite being aborted"
        assert not gen_a.done.is_set(), "A was cut short by B's disconnect"

        lib.released.set()                 # let A finish naturally
        assert gen_a.done.wait(timeout=5)
        assert gen_a.error is None
        assert gen_a.finish_reason == "stop"
    run(body())


def test_disconnect_during_our_own_query_aborts_it():
    async def body():
        lib = BlockingGenieLib()
        slot = make_slot(lib)
        gen = Generation("req-solo", slot, lib)
        start_generation(lib, slot, QueryPlan(full_prompt="hello"),
                         GenParams(max_tokens=8), gen, None, 30.0)
        assert lib.in_query.wait(timeout=5)

        gen.abort()

        assert lib.abort_signals == 1
        assert gen.done.wait(timeout=5), "the query did not unwind after the abort"
    run(body())


def test_worker_queues_the_sentinel_even_when_it_never_runs():
    """Whoever is reading gen.queue must always see the end-of-stream None,
    or a disconnected client's coroutine would hang on the queue."""
    async def body():
        lib = FakeGenieLib()
        slot = make_slot(lib)
        slot.lock.acquire()                # nobody else can get in
        try:
            gen = Generation("req-blocked", slot, lib)
            start_generation(lib, slot, QueryPlan(full_prompt="hello"),
                             GenParams(max_tokens=8), gen, None, 30.0)
            gen.abort()
            assert gen.done.wait(timeout=5)
            assert await asyncio.wait_for(gen.queue.get(), timeout=2) is None
        finally:
            slot.lock.release()
    run(body())


# ------------------------------------------------------- the SSE consumer

class FakeRequest:
    """Reports the client as gone from the Nth is_disconnected() call on."""

    def __init__(self, disconnect_after: int = 0):
        self.calls = 0
        self._after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > self._after


@pytest.mark.parametrize("abortable,expected_signals", [(True, 1), (False, 0)])
def test_sse_stream_abort_policy_on_disconnect(state, abortable, expected_signals):
    """The text path aborts on disconnect; the VLM path deliberately does not
    (GenieNode/GeniePipeline have no abort API), so it must not signal the
    text dialog either."""
    from genie_server.app import _sse_stream

    async def body():
        lib = state.lib
        slot = state.manager.slots[0]
        gen = Generation("req-sse", slot, lib)
        gen.query_active.set()             # our query is the one running

        chunks = []
        async for chunk in _sse_stream(
                FakeRequest(), state, gen, "genie-local",
                make_delta=lambda text: {"delta": text},
                make_final=lambda reason: {"finish_reason": reason},
                abortable=abortable):
            chunks.append(chunk)

        assert gen.aborted.is_set() is abortable
        assert lib.abort_signals == expected_signals
        # A disconnected client gets no final chunk and no [DONE].
        assert chunks == []
    run(body())


def test_sse_stream_completes_normally_when_the_client_stays(state):
    from genie_server.app import _sse_stream

    async def body():
        lib = state.lib
        slot = state.manager.slots[0]
        gen = Generation("req-sse-ok", slot, lib)
        gen.put_threadsafe("hello")
        gen.put_threadsafe(None)

        chunks = []
        async for chunk in _sse_stream(
                FakeRequest(disconnect_after=10), state, gen, "genie-local",
                make_delta=lambda text: {"delta": text},
                make_final=lambda reason: {"finish_reason": reason}):
            chunks.append(chunk)

        assert lib.abort_signals == 0
        assert not gen.aborted.is_set()
        assert chunks[-1] == "data: [DONE]\n\n"
        assert any("hello" in c for c in chunks)
    run(body())

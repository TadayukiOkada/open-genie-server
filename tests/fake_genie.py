"""A fake GenieLib with the same method surface as genie_server.capi.GenieLib.

Custom-sampler (logprobs) support: register_custom_sampler stores the
handler, and query() in custom mode builds a real ctypes float32 logits
buffer per step (vocab=512, planned token boosted to +50) and lets the
handler choose the emitted token — exercising the genuine
LogprobsCollector/teacher-forcing paths without an NPU. Assign a
FakeTokenizer to `lib.tokenizer` so the fake can plan token ids.

Lets the whole HTTP layer, engine, templates, and tool-calling logic run
offline (no Hexagon NPU, no libGenie.so). The scripted response depends on
the prompt so tests can exercise specific paths:

  - prompt contains "TOOLCALL"  -> response contains a <tool_call> block
  - prompt contains "MANGLED"   -> a correct call body whose <tool_call> marker
                                   came out as Cyrillic, the way
                                   qwen3_4b_instruct_2507 w4a16 does on the
                                   board (F25)
  - prompt contains "ERROR"     -> query returns a negative status
  - otherwise                   -> "Hello world from Genie!" in 5 chunks
"""

import json

from genie_server import capi


class FakeHandle:
    def __init__(self, ident: int):
        self.value = ident  # non-null, like a real ctypes handle


class FakeGenieLib:
    N_VOCAB = 512         # fake logits width (token ids must stay below this)
    PLANNED_BOOST = 50.0  # softmax(50 vs 0) ~ 1.0 => deterministic sampling

    def __init__(self):
        self.tokenizer = None       # assigned by tests (FakeTokenizer)
        self.custom_samplers = {}   # name -> on_logits handler
        self._next_handle = 1
        self.freed: list = []
        self.max_tokens: dict = {}
        self.stop_sequences: dict = {}
        self.sampler_params: dict = {}
        self.reset_count = 0
        self.abort_signals = 0
        self.applied_lora: dict = {}
        self.lora_strengths: list = []   # (handle, engine, tensor, alpha)
        self.lora_strength_status = 0    # set non-zero to fake an SDK failure
        self.fail_create = False         # set True to fake a failed model load
        self.queries = []                # every prompt query() was given
        self.canned_response = None      # set to answer with exactly this
        self.performance_policy: dict = {}
        self.saved_states: dict = {}
        self.validator_flag_resets = 0
        self.bound_profiles: list = []
        self.freed_profiles: list = []
        self.bound_loggers: list = []    # one entry per create_dialog call
        self.created_loggers: list = []  # levels passed to create_logger
        self.freed_loggers: list = []

    @property
    def cdll(self):
        raise RuntimeError("FakeGenieLib has no CDLL (VLM not supported in tests)")

    # ------------------------------------------------------------ lifecycle

    def create_dialog(self, config_json: bytes, profile=None,
                      log_handle=None) -> FakeHandle:
        if self.fail_create:
            raise RuntimeError("GenieDialog_create failed: -1")
        h = FakeHandle(self._next_handle)
        self._next_handle += 1
        self.bound_profiles.append(profile)
        self.bound_loggers.append(log_handle)
        return h

    # ------------------------------------------------------------ logging

    def create_logger(self, level: str) -> FakeHandle:
        from genie_server.capi import LOG_LEVELS
        if level not in LOG_LEVELS:
            raise ValueError(f"Unknown Genie log level {level!r}")
        self.created_loggers.append(level)
        h = FakeHandle(2000 + self._next_handle)
        self._next_handle += 1
        return h

    def free_logger(self, log_handle) -> None:
        if log_handle is not None and log_handle.value:
            self.freed_loggers.append(log_handle.value)

    # ------------------------------------------------------------ profiling

    def create_profile(self) -> FakeHandle:
        h = FakeHandle(1000 + self._next_handle)
        self._next_handle += 1
        return h

    def get_profile_json(self, profile) -> str:
        """Shaped like Profile.cpp's output: KPIs as {value, unit} under a
        per-component stat object."""
        return json.dumps({"profile": {"dialog": [{
            "type": "GenieDialog_query", "duration": 1234567,
            "time-to-first-token": {"value": 50000, "unit": "MICROSEC"},
            "prompt-processing-rate": {"value": 2700.5, "unit": "TPS"},
            "num-prompt-tokens": {"value": 33, "unit": "NONE"},
            "token-generation-rate": {"value": 68.8, "unit": "TPS"},
            "token-generation-time": {"value": 900000, "unit": "MICROSEC"},
            "num-generated-tokens": {"value": 64, "unit": "NONE"},
        }]}})

    def free_profile(self, profile) -> None:
        self.freed_profiles.append(profile)

    def reset_dialog_validator_flags(self) -> None:
        self.validator_flag_resets += 1

    def free_dialog(self, handle) -> None:
        if handle is not None and handle.value:
            self.freed.append(handle.value)

    # ------------------------------------------------------------ inference

    def _response_for(self, text: str) -> str:
        if self.canned_response is not None:
            return self.canned_response
        if "MANGLED" in text:
            return ('\u0424\u0420\u0410\u0413\u041c\u0415\u041d\u0422\n'
                    '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
                    '\u0424\u0420\u0410\u0413\u041c\u0415\u041d\u0422')
        if "TOOLCALL" in text:
            return ('I will check that.\n<tool_call>\n'
                    '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
                    '</tool_call>')
        return "Hello world from Genie!"

    def query(self, handle, text: str, sentence_code: int, on_token) -> int:
        self.queries.append(text)
        if "ERROR" in text:
            return -1
        if sentence_code == capi.SENTENCE_BEGIN:  # prefix warmup: prefill only
            on_token("", capi.SENTENCE_END)
            return 0
        params = self.sampler_params.get(handle.value, {})
        if params.get("type") == "custom":
            return self._query_custom(handle, text,
                                      params["callback-name"], on_token)
        response = self._response_for(text)
        # Split into ~word chunks like a real token stream.
        chunks = [w + " " for w in response.split(" ")]
        chunks[-1] = chunks[-1].rstrip()
        limit = self.max_tokens.get(handle.value, capi.MAX_NUM_TOKENS_UNLIMITED)
        for chunk in chunks[:limit]:
            on_token(chunk, capi.SENTENCE_CONTINUE)
        on_token("", capi.SENTENCE_END)
        return 0

    def _query_custom(self, handle, text: str, cb_name: str, on_token) -> int:
        """Per-step logits -> registered handler -> emitted token. The step
        count follows max_tokens when it is small/explicit (forced scoring,
        max_tokens caps), else the planned response length (mimicking EOS)."""
        import ctypes

        handler = self.custom_samplers[cb_name]
        planned = self.tokenizer.encode(self._response_for(text)).ids
        limit = self.max_tokens.get(handle.value, capi.MAX_NUM_TOKENS_UNLIMITED)
        steps = limit if limit <= 256 else min(limit, len(planned))
        for i in range(steps):
            logits = (ctypes.c_float * self.N_VOCAB)()
            target = planned[i] if i < len(planned) else 0
            if target >= self.N_VOCAB:  # vocab grew past the fake width
                target = 0
            logits[target] = self.PLANNED_BOOST
            ids = handler(ctypes.addressof(logits), self.N_VOCAB, 1)
            tok = int(ids[0])
            piece = self.tokenizer.decode([tok])
            on_token(piece + (" " if i < steps - 1 else ""), capi.SENTENCE_CONTINUE)
        on_token("", capi.SENTENCE_END)
        return 0

    def register_custom_sampler(self, name: str, on_logits) -> None:
        self.custom_samplers[name] = on_logits

    def reset(self, handle) -> int:
        self.reset_count += 1
        return 0

    def signal_abort(self, handle) -> int:
        self.abort_signals += 1
        return 0

    def save_state(self, handle, path: str) -> int:
        self.saved_states[path] = True
        with open(path, "wb") as f:
            f.write(b"fake-kv-state")
        return 0

    def restore_state(self, handle, path: str) -> int:
        return 0 if path in self.saved_states else -1

    # ------------------------------------------------------------ parameters

    def set_max_tokens(self, handle, max_tokens) -> None:
        self.max_tokens[handle.value] = \
            int(max_tokens) if max_tokens else capi.MAX_NUM_TOKENS_UNLIMITED

    def set_stop_sequences(self, handle, stop) -> None:
        self.stop_sequences[handle.value] = stop

    def apply_sampler_params(self, handle, params) -> None:
        self.sampler_params[handle.value] = params

    def apply_sampler_params_to_handle(self, sampler_h, params) -> None:
        pass

    # ------------------------------------------------------------ introspection

    def get_value_string(self, handle, key) -> str:
        return ""

    def get_context_occupancy(self, handle) -> int:
        return 0

    def get_applied_lora(self, handle) -> str:
        return self.applied_lora.get(handle.value, "")

    # ------------------------------------------------------------ LoRA

    def apply_lora(self, handle, engine: str, adapter: str) -> int:
        self.applied_lora[handle.value] = adapter
        return 0

    def set_lora_strength(self, handle, engine: str, tensor: str, alpha: float) -> int:
        self.lora_strengths.append((handle.value, engine, tensor, alpha))
        return self.lora_strength_status

    def release_lora_memory(self, handle, engine: str, adapter: str) -> int:
        self.applied_lora.pop(handle.value, None)
        return 0

    # ------------------------------------------------------------ performance

    def set_performance_policy(self, handle, policy_value: int) -> int:
        self.performance_policy[handle.value] = policy_value
        return 0

    def get_performance_policy(self, handle) -> int:
        return self.performance_policy.get(handle.value, 40)


class FakeTokenizer:
    """Whitespace 'tokenizer' with the two methods the server uses."""

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def __init__(self):
        self._vocab = {}
        self._rev = {}

    def encode(self, text: str):
        ids = []
        for w in text.split():
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab) + 1
                self._rev[self._vocab[w]] = w
            ids.append(self._vocab[w])
        return self._Encoding(ids)

    def decode(self, ids, skip_special_tokens=False) -> str:
        return " ".join(self._rev.get(i, f"<unk{i}>") for i in ids)

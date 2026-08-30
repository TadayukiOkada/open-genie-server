"""Token logprobs via the SDK's custom-sampler hook.

The Genie SDK exposes no logits through GenieDialog — but its custom sampler
(GenieSampler_registerUserDataCallback + sampler config {"type": "custom"})
hands the host the full dequantized float32 logits vector at every
generation step and lets the host choose the emitted token. That enables:

- **Sample mode** (OpenAI `logprobs` / `top_logprobs`): sampling moves into
  this module (greedy / temperature / top-k / top-p over the logits), and
  each step's chosen-token logprob plus the top-N alternatives is recorded.
- **Force mode** (lm_eval loglikelihood, `echo=true` + `logprobs`): the
  callback ignores the distribution for *choosing* and instead returns the
  next prompt token (teacher forcing), recording P(token_i | tokens_<i) for
  every prompt position. Exact, but runs the whole prompt at decode speed —
  which is why the server gates it behind an explicit switch (see
  /v1/server/prompt_logprobs).

Costs (sample mode): one log-softmax over the vocab per token (~1-2 ms on
target-class Arm cores with numpy) versus tens of ms of NPU decode — a
single-digit-percent overhead, and zero for requests that don't ask for
logprobs (they stay on the SDK's basic sampler).

numpy is required; without it logprobs requests are rejected upfront.
"""

import ctypes
import logging
import math

logger = logging.getLogger(__name__)

try:
    import numpy as np
    LOGPROBS_AVAILABLE = True
except ImportError:
    LOGPROBS_AVAILABLE = False


class LogprobsCollector:
    """Per-request logprobs state; on_logits() runs inside the SDK's custom
    sampler callback (inference thread). One collector per generation, wired
    up by the engine under the slot lock."""

    def __init__(self, top_n: int = 0, temperature=None, top_p=None, top_k=None,
                 seed=None, forced_tokens: list[int] | None = None,
                 status_entry: dict | None = None):
        self.top_n = int(top_n)
        self.forced_tokens = forced_tokens  # None => sample mode
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._rng = np.random.default_rng(seed)
        self.step = 0
        # Per step: (token_id, logprob, [(alt_id, alt_logprob), ...])
        self.results: list[tuple[int, float, list[tuple[int, float]]]] = []
        self._status_entry = status_entry

    # -------------------------------------------------------- sampler hook

    def on_logits(self, logits_addr, n_floats: int, num_tokens: int) -> list[int]:
        logits = np.ctypeslib.as_array(
            ctypes.cast(logits_addr, ctypes.POINTER(ctypes.c_float)), (n_floats,))

        # log-softmax normalizer (one pass; float32 in, float64 accumulate)
        m = float(logits.max())
        log_z = m + math.log(float(np.exp(logits - m, dtype=np.float64).sum()))

        if self.forced_tokens is not None:
            if self.step < len(self.forced_tokens):
                token = int(self.forced_tokens[self.step])
                if token >= n_floats:
                    logger.error(f"forced token {token} out of vocab ({n_floats})")
                    token = int(logits.argmax())
            else:  # shouldn't happen (max_tokens == len(forced_tokens))
                token = int(logits.argmax())
        else:
            token = self._sample(logits)

        top = []
        if self.top_n > 0:
            k = min(self.top_n, n_floats)
            idx = np.argpartition(logits, -k)[-k:]
            idx = idx[np.argsort(logits[idx])[::-1]]
            top = [(int(i), float(logits[i]) - log_z) for i in idx]

        self.results.append((token, float(logits[token]) - log_z, top))
        self.step += 1
        if self._status_entry is not None and self.step % 16 == 0:
            self._status_entry["detail"] = f"{self.step} tokens scored"
        return [token]

    def _sample(self, logits) -> int:
        """Greedy / temperature / top-k / top-p sampling — mirrors the SDK's
        basic sampler semantics (custom mode bypasses it entirely)."""
        temp = self.temperature
        if temp is None:
            temp = 1.0
        if temp <= 0.0 or self.top_k == 1:
            return int(logits.argmax())

        scaled = logits.astype(np.float64) / temp
        if self.top_k and self.top_k > 0:
            keep = np.argpartition(scaled, -self.top_k)[-self.top_k:]
        else:
            keep = np.arange(scaled.shape[0])
        probs = np.exp(scaled[keep] - scaled[keep].max())
        probs /= probs.sum()

        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[order])
            cut = int(np.searchsorted(cum, self.top_p) + 1)
            order = order[:cut]
            keep, probs = keep[order], probs[order]
            probs /= probs.sum()

        return int(self._rng.choice(keep, p=probs))


# ---------------------------------------------------------------- rendering

def _token_str(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:  # tokenizer without the kwarg
        return tokenizer.decode([token_id])


def completions_logprobs(tokenizer, results, first_token_id: int | None = None) -> dict:
    """OpenAI text_completion `logprobs` object. In echo/force mode,
    `first_token_id` prepends one entry with a null logprob (the first
    echoed token's probability is undefined — same as OpenAI's echo
    behavior; lm_eval's ctxlen slicing skips it)."""
    tokens, token_logprobs, top_logprobs, offsets = [], [], [], []
    offset = 0
    if first_token_id is not None:
        text = _token_str(tokenizer, first_token_id)
        tokens.append(text)
        token_logprobs.append(None)
        top_logprobs.append(None)
        offsets.append(offset)
        offset += len(text)
    for token_id, lp, top in results:
        text = _token_str(tokenizer, token_id)
        tokens.append(text)
        token_logprobs.append(round(lp, 8))
        top_logprobs.append(
            {_token_str(tokenizer, i): round(v, 8) for i, v in top} if top else {})
        offsets.append(offset)
        offset += len(text)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }


def chat_logprobs(tokenizer, results) -> dict:
    """OpenAI chat.completion `logprobs` object ({"content": [...]})."""
    content = []
    for token_id, lp, top in results:
        text = _token_str(tokenizer, token_id)
        content.append({
            "token": text,
            "logprob": round(lp, 8),
            "bytes": list(text.encode("utf-8")),
            "top_logprobs": [
                {"token": _token_str(tokenizer, i),
                 "logprob": round(v, 8),
                 "bytes": list(_token_str(tokenizer, i).encode("utf-8"))}
                for i, v in top
            ],
        })
    return {"content": content}

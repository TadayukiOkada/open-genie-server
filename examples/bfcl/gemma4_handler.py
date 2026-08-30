"""Register a gemma4 model id with BFCL.

BFCL picks the chat template from the model id, and its `GemmaHandler`
hard-codes Gemma 2/3's `<start_of_turn>` / `<end_of_turn>` — it does not read
the tokenizer's template. gemma4 marks turns with `<|turn>` / `<turn|>` (ids
105/106) and **does not have the Gemma 2/3 spelling in its vocabulary at all**,
so running it under `google/gemma-3-*` writes markers that split into roughly
nine ordinary tokens each and puts the model off its trained format.

This module subclasses the handler with the right markers and registers it as
`google/gemma4-e2b-it`. Everything else — the system-prompt preprocessing, the
role substitution, the scoring — is inherited unchanged, so a score from this
handler is comparable to one from `GemmaHandler`.

Usage (the venv's python, so it patches the same install bfcl runs from):

    /tmp/bfclvenv/bin/python gemma4_handler.py     # verify registration
    BFCL_EXTRA_HANDLER=/path/to/gemma4_handler.py ./run_bfcl.sh ...

or import it before `bfcl` starts via sitecustomize / PYTHONSTARTUP. The
simplest thing, and what we did, is to let install_gemma4_handler() write the
handler into the installed package — the venv is disposable.
"""
from bfcl_eval.model_handler.local_inference.gemma import GemmaHandler
from overrides import override


class Gemma4Handler(GemmaHandler):
    """GemmaHandler with gemma4's turn markers."""

    BEGIN_TURN = "<|turn>"
    END_TURN = "<turn|>"

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = "<bos>"

        if messages[0]["role"] == "system":
            first_user_prefix = messages[0]["content"].strip() + "\n\n"
            messages = messages[1:]
        else:
            first_user_prefix = ""

        is_first = True
        for message in messages:
            formatted_prompt += (
                f"{self.BEGIN_TURN}{message['role']}\n"
                f"{first_user_prefix if is_first else ''}"
                f"{message['content'].strip()}{self.END_TURN}\n"
            )
            is_first = False

        formatted_prompt += f"{self.BEGIN_TURN}model\n"

        return formatted_prompt


def install(model_id="google/gemma4-e2b-it", display_name="Gemma4-e2b-it (Prompt)"):
    """Adds the model id to BFCL's mapping in the running interpreter."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    MODEL_CONFIG_MAPPING[model_id] = ModelConfig(
        model_name=model_id,
        display_name=display_name,
        url="https://ai.google.dev/gemma",
        org="Google",
        license="gemma-terms-of-use",
        model_handler=Gemma4Handler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )
    try:  # keep the convenience index in step; not load-bearing
        from bfcl_eval.constants import supported_models
        if model_id not in supported_models.SUPPORTED_MODELS:
            supported_models.SUPPORTED_MODELS.append(model_id)
    except Exception:
        pass
    return model_id


if __name__ == "__main__":
    mid = install()
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    cfg = MODEL_CONFIG_MAPPING[mid]
    h = cfg.model_handler.__new__(cfg.model_handler)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    print("registered:", mid, "->", cfg.model_handler.__name__)
    print("prompt:", repr(h._format_prompt(msgs, [])))

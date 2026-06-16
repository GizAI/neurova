from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .adapter import ModelAdapter
from .features import RuntimeFeatures
from .runtime import GenerationConfig, RuntimeEngine


def assert_minimal_adapter_conformance(
    adapter: ModelAdapter,
    *,
    model_dir: Path,
    prompt: str = "ab",
    messages: Sequence[dict[str, Any]] | None = None,
    max_new_tokens: int = 2,
) -> None:
    """Run the minimum runtime contract expected from any decoder adapter.

    This is intentionally not Qwen-specific.  It verifies that a new adapter can
    load config/tokenizer/model, allocate state, encode prompt/messages, prefill,
    and generate through RuntimeEngine without server or CLI special-casing.
    """

    engine = RuntimeEngine(
        adapter=adapter,
        hf_model=model_dir,
        qb_model=model_dir,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("original"),
    )
    prompt_ids = engine.encode_prompt(prompt)
    if not prompt_ids:
        raise AssertionError("adapter encode_prompt returned no tokens")
    state = engine.new_state()
    engine.prefill(prompt_ids, state)
    generated = engine.generate_ids_greedy_gpu(
        prompt_ids,
        GenerationConfig(max_new_tokens=max_new_tokens, eos_token_ids=()),
    )
    if len(generated) != max_new_tokens:
        raise AssertionError("adapter generation did not produce the requested token count")
    msg_payload = list(messages or [{"role": "user", "content": prompt}])
    if not engine.encode_messages(msg_payload):
        raise AssertionError("adapter encode_messages returned no tokens")

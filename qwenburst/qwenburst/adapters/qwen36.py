from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ..config import Qwen36_27B_TextConfig
from ..core.adapter import AdapterDescriptor, adapter_registry
from ..core.features import RuntimeFeatures
from ..loader import QuantizedStore
from ..model import QwenBurstModel
from ..state import DecodeState


def choose_qwen_weight_device(qb_model: Path, requested: str, runtime_device: str) -> str:
    if requested != "auto":
        return runtime_device if requested == "cuda" else "cpu"
    index = json.loads((Path(qb_model) / "qwenburst_index.json").read_text(encoding="utf-8"))
    if any(meta.get("kind") == "lowbit_marlin_groupwise" for meta in index.get("tensors", {}).values()):
        return runtime_device
    bits = {
        int(meta["bits"])
        for meta in index.get("tensors", {}).values()
        if meta.get("kind") == "lowbit_symmetric_groupwise"
    }
    return runtime_device if bits and max(bits) <= 3 else "cpu"


def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(p for p in parts if p)


class Qwen36Adapter:
    descriptor = AdapterDescriptor(
        adapter_id="qwen36",
        family="qwen3.6-hybrid-gdn",
        default_model_name="qwenburst-qwen3.6-27b-q4-marlin",
        supports_state=True,
        supports_mtp=False,
    )

    def load_config(self, hf_model: Path) -> Qwen36_27B_TextConfig:
        cfg_path = Path(hf_model) / "config.json"
        return Qwen36_27B_TextConfig.from_hf_config(cfg_path) if cfg_path.exists() else Qwen36_27B_TextConfig()

    def load_tokenizer(self, hf_model: Path):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers is required for qwenburst chat/server") from exc
        return AutoTokenizer.from_pretrained(str(hf_model), trust_remote_code=True)

    def create_model(
        self,
        *,
        qb_model: Path,
        cfg: Qwen36_27B_TextConfig,
        device: str,
        weight_device: str,
        cpu_embed: bool = False,
    ) -> QwenBurstModel:
        resolved_weight_device = choose_qwen_weight_device(qb_model, weight_device, device)
        store = QuantizedStore(qb_model, device=resolved_weight_device)
        embed_store = QuantizedStore(qb_model, device="cpu") if cpu_embed else None
        return QwenBurstModel(store, cfg=cfg, device=device, embed_store=embed_store, head_store=None)

    def allocate_state(
        self,
        cfg: Qwen36_27B_TextConfig,
        *,
        recent_window: int,
        device: str,
        features: RuntimeFeatures,
    ) -> DecodeState:
        return DecodeState.allocate(
            cfg,
            max_seq_len=recent_window,
            device=device,
            kv_window_policy=features.kv_window_policy,
        )

    def encode_prompt(self, tokenizer, prompt: str, system: str | None = None) -> list[int]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.encode_messages(tokenizer, messages)

    def encode_messages(self, tokenizer, messages: Sequence[dict[str, Any]]) -> list[int]:
        payload = [{"role": m.get("role", "user"), "content": _content_to_text(m.get("content", ""))} for m in messages]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                encoded = tokenizer.apply_chat_template(payload, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                encoded = tokenizer.apply_chat_template(payload, tokenize=True, add_generation_prompt=True)
            if isinstance(encoded, dict):
                encoded = encoded["input_ids"]
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.reshape(-1).tolist()
            if encoded and isinstance(encoded[0], (list, tuple)):
                encoded = encoded[0]
            return [int(t) for t in encoded]
        text = "\n".join(str(m["content"]) for m in payload)
        return [int(t) for t in tokenizer.encode(text)]

    def eos_token_ids(self, tokenizer) -> tuple[int, ...]:
        ids = []
        for name in ("eos_token_id", "pad_token_id"):
            val = getattr(tokenizer, name, None)
            if isinstance(val, int):
                ids.append(val)
        return tuple(set(ids))


adapter_registry.register(Qwen36Adapter())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .qwen36_impl.config import Qwen36_27B_TextConfig
from ..core.adapter import AdapterDescriptor, adapter_registry
from ..core.chat_template import resolve_chat_template_kwargs
from ..core.features import RuntimeCapabilities, RuntimeFeatures
from ..core.kv_cache import KVCacheLayout, KVCacheSpec
from ..core.platform import resolve_index_file
from ..loader import QuantizedStore
from .qwen36_impl.model import Qwen36Model
from .qwen36_mtp import native_mtp1_proposer_for_model
from .qwen36_impl.state import DecodeState, DecodeStateArena


def _largest_power_of_two_divisor(value: int) -> int:
    value = max(1, int(value))
    return value & -value


def _resolved_kv_cache_spec(cfg: Qwen36_27B_TextConfig, features: RuntimeFeatures | None) -> KVCacheSpec:
    raw_dtype = features.kv_cache_dtype if features is not None else cfg.kv_cache_mode
    hadamard_order = min(128, _largest_power_of_two_divisor(cfg.attention_head_dim))
    return KVCacheSpec.resolve(raw_dtype, hadamard_order=hadamard_order)


def choose_qwen_weight_device(qb_model: Path, requested: str, runtime_device: str) -> str:
    if requested != "auto":
        return runtime_device if requested == "cuda" else "cpu"
    index = json.loads(resolve_index_file(qb_model).read_text(encoding="utf-8"))
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
        default_model_name="langburst-qwen3.6-27b-q4",
        capabilities=RuntimeCapabilities.stateful_hybrid(),
        supports_state=True,
        supports_mtp=True,
    )

    def load_config(self, hf_model: Path) -> Qwen36_27B_TextConfig:
        cfg_path = Path(hf_model) / "config.json"
        return Qwen36_27B_TextConfig.from_hf_config(cfg_path) if cfg_path.exists() else Qwen36_27B_TextConfig()

    def load_tokenizer(self, hf_model: Path):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers is required for langburst chat/server") from exc
        return AutoTokenizer.from_pretrained(str(hf_model), trust_remote_code=True)

    def create_model(
        self,
        *,
        qb_model: Path,
        cfg: Qwen36_27B_TextConfig,
        device: str,
        weight_device: str,
        cpu_embed: bool = False,
    ) -> Qwen36Model:
        resolved_weight_device = choose_qwen_weight_device(qb_model, weight_device, device)
        store = QuantizedStore(qb_model, device=resolved_weight_device)
        embed_store = QuantizedStore(qb_model, device="cpu") if cpu_embed else None
        return Qwen36Model(store, cfg=cfg, device=device, embed_store=embed_store, head_store=None)

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
            kv_cache_spec=_resolved_kv_cache_spec(cfg, features),
        )

    def estimate_state_bytes(
        self,
        cfg: Qwen36_27B_TextConfig,
        *,
        recent_window: int,
        features: RuntimeFeatures | None = None,
        ) -> int:
        dtype_bytes = 2
        kv_spec = _resolved_kv_cache_spec(cfg, features)
        gdn = (
            len(cfg.gdn_layers)
            * cfg.linear_num_value_heads
            * cfg.linear_key_head_dim
            * cfg.linear_value_head_dim
            * dtype_bytes
        )
        conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        conv = len(cfg.gdn_layers) * conv_dim * (cfg.linear_conv_kernel_dim - 1) * dtype_bytes
        kv_layout = KVCacheLayout.from_parts(
            layers=cfg.attention_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.attention_head_dim,
        )
        kv = kv_layout.total_bytes(kv_spec, int(recent_window))
        return int(gdn + conv + kv)

    def estimate_arena_state_bytes(
        self,
        cfg: Qwen36_27B_TextConfig,
        *,
        max_slots: int,
        kv_num_blocks: int,
        kv_block_size: int,
        features: RuntimeFeatures,
    ) -> int:
        dtype_bytes = 2
        kv_spec = _resolved_kv_cache_spec(cfg, features)
        gdn_per_slot = (
            len(cfg.gdn_layers)
            * cfg.linear_num_value_heads
            * cfg.linear_key_head_dim
            * cfg.linear_value_head_dim
            * dtype_bytes
        )
        conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        conv_per_slot = len(cfg.gdn_layers) * conv_dim * (cfg.linear_conv_kernel_dim - 1) * dtype_bytes
        kv_layout = KVCacheLayout.from_parts(
            layers=cfg.attention_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.attention_head_dim,
        )
        paged_kv = kv_layout.total_bytes(kv_spec, int(kv_num_blocks) * int(kv_block_size))
        return int((gdn_per_slot + conv_per_slot) * int(max_slots) + paged_kv)

    def create_state_arena(
        self,
        cfg: Qwen36_27B_TextConfig,
        *,
        max_seq_len: int,
        num_slots: int,
        device: str,
        features: RuntimeFeatures,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
    ) -> DecodeStateArena:
        return DecodeStateArena(
            cfg=cfg,
            max_seq_len=max_seq_len,
            num_slots=num_slots,
            kv_num_blocks=kv_num_blocks,
            kv_block_size=kv_block_size,
            device=device,
            kv_window_policy=features.kv_window_policy,
            kv_cache_spec=_resolved_kv_cache_spec(cfg, features),
        )

    def encode_prompt(self, tokenizer, prompt: str, system: str | None = None) -> list[int]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.encode_messages(tokenizer, messages)

    def encode_messages(self, tokenizer, messages: Sequence[dict[str, Any]], **kwargs: Any) -> list[int]:
        payload = [{"role": m.get("role", "user"), "content": _content_to_text(m.get("content", ""))} for m in messages]
        if hasattr(tokenizer, "apply_chat_template"):
            chat_template_kwargs = resolve_chat_template_kwargs(
                base=kwargs.get("chat_template_kwargs") or None
            )
            try:
                encoded = tokenizer.apply_chat_template(
                    payload,
                    tokenize=True,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
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
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            return (int(eos),)
        pad = getattr(tokenizer, "pad_token_id", None)
        return (int(pad),) if isinstance(pad, int) else ()

    def create_speculative_proposer(self, model: Qwen36Model):
        return native_mtp1_proposer_for_model(model)


class Qwen36A3BAdapter(Qwen36Adapter):
    descriptor = AdapterDescriptor(
        adapter_id="qwen36-a3b",
        family="qwen3.6-hybrid-a3b",
        default_model_name="langburst-qwen3.6-35b-a3b-q4",
        capabilities=RuntimeCapabilities.stateful_hybrid(),
        supports_state=True,
        supports_mtp=True,
    )


adapter_registry.register(Qwen36Adapter())
adapter_registry.register(Qwen36A3BAdapter())

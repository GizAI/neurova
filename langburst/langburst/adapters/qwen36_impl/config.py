from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Qwen36_27B_TextConfig:
    """Text-only language-model shape of Qwen3.6-27B.

    Values are taken from the public model card.  The engine intentionally
    ignores the vision encoder for the 16GB target.
    """

    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_layers: int = 64
    vocab_size_padded: int = 248_320
    # 16 x (3 GatedDeltaNet -> 1 GatedAttention)
    cycle: tuple[str, str, str, str] = ("gdn", "gdn", "gdn", "attn")
    # Optional explicit per-layer layout imported from HF configs.
    layer_types: tuple[str, ...] | None = None

    # Gated DeltaNet
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Full attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    attention_head_dim: int = 256
    rope_dim: int = 64
    rope_theta: float = 1_000_000.0

    rms_norm_eps: float = 1e-6

    # Default engine choices for 16GB Ada.
    group_size: int = 128
    weight_mode: Literal["q4", "q3q4-hybrid"] = "q4"
    kv_cache_mode: Literal["fp16", "q8", "q4"] = "fp16"
    mtp_max_steps: int = 4

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "Qwen36_27B_TextConfig":
        """Best-effort import of shape constants from a Hugging Face config.json.

        The public Qwen3.6 checkpoints may evolve slightly.  This loader keeps
        LangBurst from silently assuming stale dimensions when config.json is
        available. Unknown fields intentionally fall back to the hard-coded
        Qwen3.6-27B text defaults.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("text_config"), dict):
            data = data["text_config"]

        def pick(*names, default):
            for n in names:
                if n in data:
                    return data[n]
            return default

        base = cls()
        rope_parameters = data.get("rope_parameters") if isinstance(data.get("rope_parameters"), dict) else {}
        head_dim = int(pick("head_dim", "attention_head_dim", default=base.attention_head_dim))
        if "rope_dim" in data:
            rope_dim = int(data["rope_dim"])
        elif "partial_rotary_factor" in data:
            rope_dim = int(head_dim * float(data["partial_rotary_factor"]))
        elif "partial_rotary_factor" in rope_parameters:
            rope_dim = int(head_dim * float(rope_parameters["partial_rotary_factor"]))
        else:
            rope_dim = base.rope_dim
        raw_layer_types = data.get("layer_types") or data.get("layers_block_type") or data.get("layer_type")
        layer_types = None
        if isinstance(raw_layer_types, list):
            normed = []
            for t in raw_layer_types:
                st = str(t).lower()
                if "linear" in st or "delta" in st or "gdn" in st:
                    normed.append("gdn")
                elif "attn" in st or "attention" in st:
                    normed.append("attn")
                else:
                    normed.append(st)
            layer_types = tuple(normed)

        return cls(
            hidden_size=int(pick("hidden_size", default=base.hidden_size)),
            intermediate_size=int(pick("intermediate_size", default=base.intermediate_size)),
            num_layers=int(pick("num_hidden_layers", "num_layers", default=base.num_layers)),
            vocab_size_padded=int(pick("vocab_size", "vocab_size_padded", default=base.vocab_size_padded)),
            linear_num_value_heads=int(pick("linear_num_value_heads", default=base.linear_num_value_heads)),
            linear_num_key_heads=int(pick("linear_num_key_heads", default=base.linear_num_key_heads)),
            linear_key_head_dim=int(pick("linear_key_head_dim", default=base.linear_key_head_dim)),
            linear_value_head_dim=int(pick("linear_value_head_dim", default=base.linear_value_head_dim)),
            linear_conv_kernel_dim=int(pick("linear_conv_kernel_dim", "conv_kernel", default=base.linear_conv_kernel_dim)),
            num_attention_heads=int(pick("num_attention_heads", default=base.num_attention_heads)),
            num_key_value_heads=int(pick("num_key_value_heads", default=base.num_key_value_heads)),
            attention_head_dim=head_dim,
            rope_dim=rope_dim,
            rope_theta=float(data.get("rope_theta", rope_parameters.get("rope_theta", base.rope_theta))),
            rms_norm_eps=float(pick("rms_norm_eps", default=base.rms_norm_eps)),
            group_size=base.group_size,
            weight_mode=base.weight_mode,
            kv_cache_mode=base.kv_cache_mode,
            mtp_max_steps=int(pick("mtp_max_steps", "num_nextn_predict_layers", default=base.mtp_max_steps)),
            layer_types=layer_types,
        )

    def layer_type(self, idx: int) -> str:
        if self.layer_types is not None and idx < len(self.layer_types):
            return self.layer_types[idx]
        return self.cycle[idx % len(self.cycle)]

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.layer_type(i) == "gdn"]

    @property
    def attention_layers(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.layer_type(i) == "attn"]

    @property
    def gdn_state_values(self) -> int:
        return (
            len(self.gdn_layers)
            * self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
        )

    @property
    def gdn_state_mib_fp16(self) -> float:
        return self.gdn_state_values * 2 / 1024**2

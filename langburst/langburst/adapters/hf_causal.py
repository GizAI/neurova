from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ..core.adapter import AdapterDescriptor
from ..core.features import RuntimeCapabilities, RuntimeFeatures
from ..core.messages import normalize_for_chat_template
from ..core.platform import env, env_flag
from ..core.tool_calls import normalize_tools_for_chat_template


def _clone_cache_obj(obj: Any) -> Any:
    """Clone a Hugging Face cache object without depending on one cache class.

    Transformers has changed cache representations several times.  Keep this
    helper structural: tensors are cloned, Python containers are recursively
    cloned, and opaque cache classes fall back to deepcopy.
    """

    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_clone_cache_obj(v) for v in obj)
    if isinstance(obj, list):
        return [_clone_cache_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clone_cache_obj(v) for k, v in obj.items()}
    to_legacy_cache = getattr(obj, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(obj), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        legacy = _clone_cache_obj(to_legacy_cache())
        return from_legacy_cache(legacy)
    return copy.deepcopy(obj)


@dataclass
class HFCausalWriteSnapshot:
    past_key_values: Any
    pos: int

    def restore_(self, state: "HFCausalState") -> None:
        state.past_key_values = _clone_cache_obj(self.past_key_values)
        state.pos = int(self.pos)


@dataclass
class HFCausalState:
    past_key_values: Any = None
    pos: int = 0

    def reset(self, *, reset_attention: bool = True) -> None:
        if reset_attention:
            self.past_key_values = None
        self.pos = 0

    def fork(self, *, clone_attention: bool = True) -> "HFCausalState":
        return HFCausalState(
            past_key_values=_clone_cache_obj(self.past_key_values) if clone_attention else self.past_key_values,
            pos=self.pos,
        )

    def copy_from_(self, other: "HFCausalState", *, copy_attention: bool = True) -> None:
        if copy_attention:
            self.past_key_values = _clone_cache_obj(other.past_key_values)
        self.pos = int(other.pos)

    def speculative_write_snapshot(self, num_tokens: int) -> HFCausalWriteSnapshot:
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        return HFCausalWriteSnapshot(past_key_values=_clone_cache_obj(self.past_key_values), pos=self.pos)

    def save_snapshot(self, path: str | Path, *, include_attention_kv: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "family": "hf-causal-decoder",
                "pos": int(self.pos),
                "include_attention_kv": bool(include_attention_kv),
                "past_key_values": _clone_cache_obj(self.past_key_values) if include_attention_kv else None,
            },
            path,
        )

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "HFCausalState":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported HF causal snapshot schema: {payload.get('schema_version')}")
        return cls(past_key_values=payload.get("past_key_values"), pos=int(payload.get("pos", 0)))


class HFCausalModelWrapper:
    """Minimal Transformers causal-LM wrapper for adapter bring-up.

    Optimized low-bit adapters should replace this for production.  The wrapper
    is still useful as a real conformance path for Gemma/Llama-style models
    because it exercises the same RuntimeEngine/generation/server contract.
    """

    def __init__(self, model: Any, *, device: str) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def forward_one(self, token: int | torch.Tensor, state: HFCausalState, *, use_mtp: bool = False, return_logits: bool = True):
        if torch.is_tensor(token):
            token_id = int(token.item())
        else:
            token_id = int(token)
        input_ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        out = self.model(input_ids=input_ids, past_key_values=state.past_key_values, use_cache=True)
        state.past_key_values = out.past_key_values
        state.pos += 1
        return out.logits[0, -1]

    @torch.no_grad()
    def forward_block(self, tokens: Sequence[int] | torch.Tensor, state: HFCausalState, *, return_logits: bool = True, logits_mode: str = "all", commit: bool = True):
        if torch.is_tensor(tokens):
            input_ids = tokens.to(device=self.device, dtype=torch.long).reshape(1, -1)
        else:
            input_ids = torch.tensor([list(map(int, tokens))], device=self.device, dtype=torch.long)
        out = self.model(input_ids=input_ids, past_key_values=state.past_key_values, use_cache=True)
        if commit:
            state.past_key_values = out.past_key_values
            state.pos += int(input_ids.numel())
        logits_rows: list[torch.Tensor] = []
        if return_logits:
            if logits_mode == "last":
                logits_rows = [out.logits[0, -1]]
            else:
                logits_rows = [row for row in out.logits[0]]
        return type("HFCausalBlockOutput", (), {"logits": logits_rows, "state": state, "hidden_taps": [], "raw_hiddens": []})()


def _torch_dtype_from_env(device: str) -> torch.dtype:
    raw = (env("HF_TORCH_DTYPE") or "auto").strip().lower()
    if raw in {"auto", ""}:
        return torch.bfloat16 if torch.cuda.is_available() and str(device).startswith("cuda") else torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[raw]
    except KeyError as exc:
        known = ", ".join(sorted(mapping)) + ", auto"
        raise ValueError(f"unknown LANGBURST_HF_TORCH_DTYPE={raw!r}; expected one of {known}") from exc


def _hf_from_pretrained_kwargs(device: str) -> tuple[dict[str, Any], bool]:
    """Resolve generic HF loader knobs from LangBurst env vars.

    This keeps CLI/server model-family agnostic while still allowing Gemma4 12B
    to fit on 16GB-class GPUs via bitsandbytes when installed.
    """

    dtype = _torch_dtype_from_env(device)
    kwargs: dict[str, Any] = {
        "trust_remote_code": env_flag("HF_TRUST_REMOTE_CODE", True),
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    attn_impl = env("HF_ATTENTION_IMPLEMENTATION")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    load_4bit = env_flag("HF_LOAD_IN_4BIT", False)
    load_8bit = env_flag("HF_LOAD_IN_8BIT", False)
    if load_4bit and load_8bit:
        raise ValueError("only one of LANGBURST_HF_LOAD_IN_4BIT or LANGBURST_HF_LOAD_IN_8BIT may be enabled")

    uses_device_map = False
    if load_4bit or load_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                "HF low-bit loading requested but transformers BitsAndBytesConfig is unavailable; "
                "install compatible transformers/bitsandbytes or disable LANGBURST_HF_LOAD_IN_4BIT/8BIT"
            ) from exc
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype if dtype in {torch.float16, torch.bfloat16} else torch.bfloat16,
                bnb_4bit_quant_type=env("HF_BNB_4BIT_QUANT_TYPE", "nf4"),
                bnb_4bit_use_double_quant=env_flag("HF_BNB_4BIT_USE_DOUBLE_QUANT", True),
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = env("HF_DEVICE_MAP", "auto")
        uses_device_map = True
    elif device_map := env("HF_DEVICE_MAP"):
        kwargs["device_map"] = device_map
        uses_device_map = True
    return kwargs, uses_device_map


def _model_load_dir(path: Path) -> Path:
    """Return a HF-loadable model directory.

    Some local converted checkpoints keep a single safetensors file with a
    descriptive filename instead of the canonical `model.safetensors`.  Build a
    symlink-only alias directory so Transformers can load it without mutating or
    copying the original checkpoint.
    """

    expected = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    if any((path / name).exists() for name in expected):
        return path
    safetensors = sorted(path.glob("*.safetensors"))
    if len(safetensors) != 1:
        return path

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    alias_root = Path(os.environ.get("LANGBURST_HF_ALIAS_CACHE", Path.home() / ".cache" / "langburst" / "hf_aliases"))
    alias = alias_root / digest
    alias.mkdir(parents=True, exist_ok=True)
    for src in path.iterdir():
        if not src.is_file():
            continue
        dst_name = "model.safetensors" if src == safetensors[0] else src.name
        dst = alias / dst_name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            dst.unlink()
        dst.symlink_to(src)
    return alias


class HFAutoCausalAdapter:
    descriptor = AdapterDescriptor(
        adapter_id="hf-auto",
        family="hf-causal-decoder",
        default_model_name="hf-causal-model",
        capabilities=RuntimeCapabilities.transformer_decoder(),
        supports_state=True,
        supports_mtp=False,
    )

    def load_config(self, hf_model: Path) -> Any:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(str(hf_model), trust_remote_code=True)

    def load_tokenizer(self, hf_model: Path) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(hf_model), trust_remote_code=True)

    def create_model(
        self,
        *,
        qb_model: Path,
        cfg: Any,
        device: str,
        weight_device: str,
        cpu_embed: bool = False,
    ) -> HFCausalModelWrapper:
        from transformers import AutoModelForCausalLM

        kwargs, uses_device_map = _hf_from_pretrained_kwargs(device)
        model = AutoModelForCausalLM.from_pretrained(
            str(_model_load_dir(Path(qb_model))),
            **kwargs,
        )
        if not uses_device_map:
            model.to(device)
        return HFCausalModelWrapper(model, device=device)

    def allocate_state(self, cfg: Any, *, recent_window: int, device: str, features: RuntimeFeatures) -> HFCausalState:
        return HFCausalState()

    def create_state_arena(
        self,
        cfg: Any,
        *,
        max_seq_len: int,
        num_slots: int,
        device: str,
        features: RuntimeFeatures,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
    ) -> None:
        return None

    def encode_prompt(self, tokenizer: Any, prompt: str, system: str | None = None) -> list[int]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.encode_messages(tokenizer, messages)

    def encode_messages(self, tokenizer: Any, messages: Sequence[dict[str, Any]], **kwargs: Any) -> list[int]:
        payload = normalize_for_chat_template(messages)
        if hasattr(tokenizer, "apply_chat_template"):
            template_args: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
            tools = normalize_tools_for_chat_template(kwargs.get("tools"))
            if tools:
                template_args["tools"] = tools
            encoded = tokenizer.apply_chat_template(payload, **template_args)
            if isinstance(encoded, torch.Tensor):
                return [int(x) for x in encoded.reshape(-1).tolist()]
            if encoded and isinstance(encoded[0], (list, tuple)):
                encoded = encoded[0]
            return [int(x) for x in encoded]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in payload) + "\nassistant:"
        return [int(x) for x in tokenizer.encode(text)]

    def eos_token_ids(self, tokenizer: Any) -> tuple[int, ...]:
        ids = []
        for name in ("eos_token_id", "pad_token_id"):
            val = getattr(tokenizer, name, None)
            if isinstance(val, int):
                ids.append(val)
        return tuple(set(ids))

    def create_speculative_proposer(self, model: Any) -> None:
        return None


class Gemma4Adapter(HFAutoCausalAdapter):
    descriptor = AdapterDescriptor(
        adapter_id="gemma4",
        family="gemma4-transformer",
        default_model_name="gemma4-12b",
        capabilities=RuntimeCapabilities.transformer_decoder(),
        supports_state=True,
        supports_mtp=False,
    )

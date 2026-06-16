from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ...loader import FP16Tensor, LowBitMarlinTensor, LowBitTensor, QuantizedStore


DEFAULT_AUX_REGEX = r"^(lm_head|model\.language_model\.embed_tokens|model\.language_model\.layers\.(0|1)\.mlp\.gate_up_proj)\.weight$"
DEFAULT_MTP_AUX_REGEX = (
    r"^(lm_head|model\.language_model\.embed_tokens|"
    r"model\.language_model\.layers\.([0-9]|1[0-1])\.mlp\.gate_up_proj|"
    r"mtp\.layers\.0\.(mlp\.(gate_up_proj|down_proj)|self_attn\.(qkv_proj|o_proj)))\.weight$"
)


@dataclass(frozen=True)
class LowBitRuntimeOptions:
    """Resolved process-local knobs for LangBurst low-bit vLLM integration."""

    aux_device: str | None
    aux_regex: str
    preload_mtp: bool
    preload_visual: bool
    preload_exclude_regex: str | None

    @classmethod
    def from_env(cls) -> "LowBitRuntimeOptions":
        preload_mtp = _env_enabled("LANGBURST_VLLM_LOWBIT_PRELOAD_MTP")
        default_aux_regex = DEFAULT_MTP_AUX_REGEX if preload_mtp else DEFAULT_AUX_REGEX
        return cls(
            aux_device=os.environ.get("LANGBURST_VLLM_LOWBIT_AUX_DEVICE"),
            aux_regex=os.environ.get("LANGBURST_VLLM_LOWBIT_AUX_REGEX", default_aux_regex),
            preload_mtp=preload_mtp,
            preload_visual=_env_enabled("LANGBURST_VLLM_LOWBIT_PRELOAD_VISUAL"),
            preload_exclude_regex=os.environ.get("LANGBURST_VLLM_LOWBIT_PRELOAD_EXCLUDE_REGEX"),
        )

    def aux_device_for_tensor(self, name: str) -> torch.device | None:
        if not self.aux_device or not self.aux_regex or not re.search(self.aux_regex, name):
            return None
        return torch.device(self.aux_device)

    def preload_name_allowed(self, name: str, *, device: torch.device | None = None) -> bool:
        aux_device = self.aux_device_for_tensor(name)
        if aux_device is not None and device is not None and str(aux_device) != str(device):
            return False
        if aux_device is None and device is not None and self.aux_device and str(device) == self.aux_device:
            return False
        if self.preload_exclude_regex and re.search(self.preload_exclude_regex, name):
            return False
        if name.startswith("visual.") or name.startswith("model.visual."):
            return self.preload_visual
        if name.startswith("mtp.") or ".mtp." in name:
            return self.preload_mtp
        return True


def _runtime_options() -> LowBitRuntimeOptions:
    return LowBitRuntimeOptions.from_env()


def _tensor_nbytes(tensor: LowBitTensor | LowBitMarlinTensor | FP16Tensor) -> int:
    if isinstance(tensor, FP16Tensor):
        return int(tensor.value.numel() * tensor.value.element_size())
    total = int(tensor.qweight.numel() * tensor.qweight.element_size())
    total += int(tensor.scales.numel() * tensor.scales.element_size())
    if isinstance(tensor, LowBitMarlinTensor):
        total += sum(int(out.numel() * out.element_size()) for out in tensor._out_cache.values())
        if tensor._workspace is not None:
            total += int(tensor._workspace.numel() * tensor._workspace.element_size())
    return total


def _default_cache_limit_bytes() -> int:
    raw = os.environ.get("LANGBURST_VLLM_LOWBIT_CACHE_MAX_GB", "6.0")
    try:
        gb = float(raw)
    except ValueError:
        gb = 6.0
    return max(0, int(gb * 1024**3))


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "true", "True"}


def _preload_enabled() -> bool:
    return _env_enabled("LANGBURST_VLLM_LOWBIT_GPU_ONLY") or _env_enabled("LANGBURST_VLLM_LOWBIT_PRELOAD")


def _aux_device_for_tensor(name: str) -> torch.device | None:
    return _runtime_options().aux_device_for_tensor(name)


def _return_to_original_device(tensor: torch.Tensor, original_device: torch.device) -> torch.Tensor:
    if tensor.device == original_device:
        return tensor.contiguous() if not tensor.is_contiguous() else tensor
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    return tensor.to(device=original_device, non_blocking=False).contiguous()


def _preload_name_allowed(name: str, *, device: torch.device | None = None) -> bool:
    return _runtime_options().preload_name_allowed(name, device=device)


class _LangBurstLowBitRuntimeCache:
    """One bounded GPU tensor cache shared by all vLLM quant methods.

    vLLM owns the Qwen/GDN model graph. This cache is only the LangBurst
    low-bit weight materialization boundary for checkpoints that vLLM cannot
    load natively yet.
    """

    def __init__(self, qb_model: Path, device: torch.device) -> None:
        os.environ.setdefault("LANGBURST_MARLIN_OUT_CACHE_POLICY", "decode_only")
        self.qb_model = qb_model
        self.device = device
        self.store = QuantizedStore(qb_model, device=device)
        self.limit_bytes = _default_cache_limit_bytes()
        self.gpu_only = _env_enabled("LANGBURST_VLLM_LOWBIT_GPU_ONLY")
        if self.gpu_only:
            self.limit_bytes = 0
        self.cache_policy = os.environ.get("LANGBURST_VLLM_LOWBIT_CACHE_POLICY", "resident").strip().lower()
        if self.cache_policy in {"static", "pinned", "pin"}:
            self.cache_policy = "resident"
        self._sizes: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._resident: set[str] = set()
        self._transient: set[str] = set()
        self._cached_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._peak_cached_bytes = 0
        self._apply_calls = 0
        self._stats_interval = _env_int("LANGBURST_VLLM_LOWBIT_STATS_INTERVAL")
        self._preload_complete = False
        if _preload_enabled():
            self._preload_gpu_tensors()
            self._preload_complete = True

    def tensor(self, name: str) -> LowBitTensor | LowBitMarlinTensor | FP16Tensor:
        before = name in self.store.cache
        if self.gpu_only and self._preload_complete and not before:
            raise RuntimeError(
                "LangBurst vLLM GPU-only low-bit mode attempted a CPU-backed lazy tensor load: "
                f"{name}. Include it in the GPU preload set or disable LANGBURST_VLLM_LOWBIT_GPU_ONLY."
            )
        tensor = self.store.tensor(name)
        if before:
            self._hits += 1
        else:
            self._misses += 1
            size = _tensor_nbytes(tensor)
            self._sizes[name] = size
            self._cached_bytes += size
            self._peak_cached_bytes = max(self._peak_cached_bytes, self._cached_bytes)
            if self.cache_policy == "resident":
                if self.limit_bytes <= 0 or self._cached_bytes <= self.limit_bytes:
                    self._resident.add(name)
                else:
                    self._transient.add(name)
        self._lru[name] = None
        self._lru.move_to_end(name)
        if self.cache_policy != "resident":
            self._evict_to_limit(protected=name)
        return tensor

    def release_after_apply(self, name: str) -> None:
        if not _env_enabled("LANGBURST_VLLM_LOWBIT_EVICT_AFTER_APPLY"):
            if self.cache_policy == "resident":
                if name in self._transient:
                    self._evict(name)
            else:
                self._evict_to_limit()
            self._apply_calls += 1
            self._maybe_print_stats()
            return
        self._evict(name)
        self.empty_cuda_cache()
        self._apply_calls += 1
        self._maybe_print_stats()

    def _evict_to_limit(self, protected: str | None = None) -> None:
        if self.limit_bytes <= 0:
            return
        while self._cached_bytes > self.limit_bytes and self._lru:
            victim = next(iter(self._lru))
            if victim == protected and len(self._lru) == 1:
                break
            if victim == protected:
                self._lru.move_to_end(victim)
                continue
            self._evict(victim)
        if self.device.type == "cuda" and self._cached_bytes > self.limit_bytes:
            self.empty_cuda_cache()

    def _evict(self, name: str) -> None:
        self.store.cache.pop(name, None)
        self._lru.pop(name, None)
        self._resident.discard(name)
        self._transient.discard(name)
        self._cached_bytes -= self._sizes.pop(name, 0)
        self._evictions += 1
        if self._cached_bytes < 0:
            self._cached_bytes = 0

    def empty_cuda_cache(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _maybe_print_stats(self) -> None:
        if self._stats_interval <= 0 or self._apply_calls % self._stats_interval:
            return
        print(
            "[langburst_lowbit][stats] "
            f"apply={self._apply_calls} hits={self._hits} misses={self._misses} "
            f"evictions={self._evictions} cached_gb={self._cached_bytes / 1024**3:.2f} "
            f"peak_gb={self._peak_cached_bytes / 1024**3:.2f} limit_gb={self.limit_bytes / 1024**3:.2f} "
            f"resident={len(self._resident)} transient={len(self._transient)} policy={self.cache_policy}",
            flush=True,
        )

    def _preload_gpu_tensors(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("LangBurst vLLM low-bit GPU preload requires a CUDA device")
        loaded = 0
        skipped = 0
        for name in self.store.index.get("tensors", {}):
            if not _preload_name_allowed(name, device=self.device):
                skipped += 1
                continue
            if name in self.store.cache:
                continue
            tensor = self.store.tensor(name)
            size = _tensor_nbytes(tensor)
            self._sizes[name] = size
            self._cached_bytes += size
            self._peak_cached_bytes = max(self._peak_cached_bytes, self._cached_bytes)
            self._resident.add(name)
            self._lru[name] = None
            self._lru.move_to_end(name)
            loaded += 1
        print(
            "[langburst_lowbit][preload] "
            f"loaded={loaded} skipped={skipped} cached_gb={self._cached_bytes / 1024**3:.2f} "
            f"gpu_only={self.gpu_only} policy={self.cache_policy}",
            flush=True,
        )


_RUNTIME_CACHES: dict[tuple[str, str], _LangBurstLowBitRuntimeCache] = {}
_AUDITED_MISSING: set[str] = set()


def _runtime_cache(qb_model: Path, device: torch.device, *, preload: bool | None = None) -> _LangBurstLowBitRuntimeCache:
    key = (str(qb_model.resolve()), str(device))
    cache = _RUNTIME_CACHES.get(key)
    if cache is None:
        old_preload = os.environ.get("LANGBURST_VLLM_LOWBIT_PRELOAD")
        old_gpu_only = os.environ.get("LANGBURST_VLLM_LOWBIT_GPU_ONLY")
        if preload is False:
            os.environ["LANGBURST_VLLM_LOWBIT_PRELOAD"] = "0"
            os.environ["LANGBURST_VLLM_LOWBIT_GPU_ONLY"] = "0"
        cache = _LangBurstLowBitRuntimeCache(qb_model, device)
        if preload is False:
            if old_preload is None:
                os.environ.pop("LANGBURST_VLLM_LOWBIT_PRELOAD", None)
            else:
                os.environ["LANGBURST_VLLM_LOWBIT_PRELOAD"] = old_preload
            if old_gpu_only is None:
                os.environ.pop("LANGBURST_VLLM_LOWBIT_GPU_ONLY", None)
            else:
                os.environ["LANGBURST_VLLM_LOWBIT_GPU_ONLY"] = old_gpu_only
        _RUNTIME_CACHES[key] = cache
    return cache


def _qb_model_from_config(config: dict[str, Any] | None = None) -> Path:
    config = config or {}
    raw = (
        config.get("qb_model")
        or config.get("langburst_qb_model")
        or os.environ.get("LANGBURST_VLLM_QB_MODEL")
    )
    if not raw:
        raise RuntimeError(
            "LangBurst vLLM low-bit mode requires qb_model. "
            "Set LANGBURST_VLLM_QB_MODEL or pass model_loader_extra_config.qb_model."
        )
    return Path(str(raw)).expanduser()


def _candidate_weight_names(prefix: str) -> tuple[str, ...]:
    base = f"{prefix}.weight"
    candidates = [
        base,
        base.replace("visual.", "model.visual.", 1),
        base.replace("model.language_model.model.", "model.language_model."),
        base.replace("language_model.model.", "language_model."),
        base.replace("language_model.model.", "model.language_model."),
        base.replace("language_model.", "model.language_model.", 1),
        base.replace("model.mtp.embed_tokens.", "model.language_model.embed_tokens.", 1),
        base.replace("mtp.embed_tokens.", "model.language_model.embed_tokens.", 1),
        base.replace("model.embed_tokens.", "model.language_model.embed_tokens.", 1),
        base.replace("model.fc.", "mtp.fc.", 1),
        base.replace("model.layers.", "mtp.layers.", 1),
        base.replace("model.norm.", "mtp.norm.", 1),
        base.replace("model.pre_fc_norm_embedding.", "mtp.pre_fc_norm_embedding.", 1),
        base.replace("model.pre_fc_norm_hidden.", "mtp.pre_fc_norm_hidden.", 1),
        base.replace("language_model.model.mtp.embed_tokens.", "model.language_model.embed_tokens.", 1),
        base.replace("language_model.lm_head.", "lm_head.", 1),
        base.replace("model.language_model.lm_head.", "lm_head.", 1),
        base.replace("model.model.", "model."),
    ]
    out: list[str] = []
    for item in candidates:
        if item not in out:
            out.append(item)
    return tuple(out)


class _LangBurstLowBitMixin:
    def __init__(self, *, tensor_name: str, qb_model: Path) -> None:
        self.tensor_name = tensor_name
        self.qb_model = qb_model

    def _cache(self, device: torch.device) -> _LangBurstLowBitRuntimeCache:
        return _runtime_cache(self.qb_model, device, preload=False)

    def _tensor(self, device: torch.device) -> LowBitTensor | LowBitMarlinTensor | FP16Tensor:
        return self._cache(device).tensor(self.tensor_name)

    def _release_after_apply(self, device: torch.device) -> None:
        self._cache(device).release_after_apply(self.tensor_name)


class LangBurstLowBitLinearMethod(_LangBurstLowBitMixin):
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        layer.langburst_lowbit_tensor_name = self.tensor_name

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        original_device = x.device
        flat = x.reshape(-1, x.shape[-1])
        if flat.dtype != torch.float16:
            flat = flat.to(dtype=torch.float16)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        aux_device = _aux_device_for_tensor(self.tensor_name)
        if aux_device is not None and aux_device != flat.device:
            flat = flat.to(device=aux_device, non_blocking=True)
        weight = self._tensor(flat.device)
        if isinstance(weight, FP16Tensor):
            out = torch.matmul(flat, weight.value.to(device=flat.device, dtype=flat.dtype).t())
        else:
            out = weight.gemm(flat)
        if bias is not None:
            out = out + bias.to(device=out.device, dtype=out.dtype)
        result = out.reshape(*original_shape, out.shape[-1])
        result = _return_to_original_device(result, original_device)
        if result.dtype != original_dtype:
            result = result.to(dtype=original_dtype)
        self._release_after_apply(flat.device)
        return result


class LangBurstLowBitEmbeddingMethod(_LangBurstLowBitMixin):
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        layer.langburst_lowbit_tensor_name = self.tensor_name

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        original_dtype = x.dtype
        original_device = x.device
        x_fp16 = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
        aux_device = _aux_device_for_tensor(self.tensor_name)
        if aux_device is not None and aux_device != x_fp16.device:
            x_fp16 = x_fp16.to(device=aux_device, non_blocking=True)
        weight = self._tensor(x_fp16.device)
        if isinstance(weight, FP16Tensor):
            out = torch.matmul(x_fp16, weight.value.to(device=x_fp16.device, dtype=x_fp16.dtype).t())
        elif isinstance(weight, LowBitMarlinTensor):
            flat = x_fp16.reshape(-1, x.shape[-1])
            if not flat.is_contiguous():
                flat = flat.contiguous()
            out = weight.gemm(flat).reshape(*x.shape[:-1], -1)
        else:
            flat = x_fp16.reshape(-1, x.shape[-1])
            if not flat.is_contiguous():
                flat = flat.contiguous()
            out = weight.gemm(flat).reshape(*x.shape[:-1], -1)
        if bias is not None:
            out = out + bias.to(device=out.device, dtype=out.dtype)
        out = _return_to_original_device(out, original_device)
        result = out if out.dtype == original_dtype else out.to(dtype=original_dtype)
        self._release_after_apply(x_fp16.device)
        return result

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        original_device = input_.device
        aux_device = _aux_device_for_tensor(self.tensor_name)
        lookup_device = aux_device or original_device
        input_on_device = input_.to(device=lookup_device, non_blocking=True) if lookup_device != original_device else input_
        weight = self._tensor(lookup_device)
        if isinstance(weight, FP16Tensor):
            out = F.embedding(input_on_device, weight.value.to(device=lookup_device))
            self._release_after_apply(lookup_device)
            return _return_to_original_device(out, original_device)
        if isinstance(weight, LowBitMarlinTensor):
            raise RuntimeError("LangBurst Marlin tensor cannot serve embedding rows")
        flat = input_on_device.reshape(-1).to(torch.long)
        rows = [weight.row_dequant(tok).to(device=lookup_device) for tok in flat]
        out = torch.stack(rows, dim=0).reshape(*input_.shape, weight.cols)
        self._release_after_apply(lookup_device)
        return _return_to_original_device(out, original_device)

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        layer.quant_method = embed_tokens.quant_method
        return layer


class LangBurstLowBitConfig:  # vLLM QuantizationConfig subclass at runtime.
    def __init__(self, qb_model: Path, full_config: dict[str, Any] | None = None) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

        if not isinstance(self, QuantizationConfig):
            QuantizationConfig.__init__(self)
        self.qb_model = qb_model
        self.full_config = full_config or {}
        self._index = QuantizedStore(qb_model, device="cpu").index

    def get_name(self) -> str:
        return "langburst_lowbit"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["langburst_lowbit_config.json", "langburst_index.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LangBurstLowBitConfig":
        return cls(_qb_model_from_config(config), config)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Any:
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )

        is_embedding = isinstance(layer, VocabParallelEmbedding)
        is_linear = isinstance(layer, (LinearBase, ParallelLMHead))
        tensor_name = self._resolve_tensor_name(prefix)
        if tensor_name is None:
            if is_embedding or is_linear:
                self._audit_missing(layer, prefix)
            if is_embedding:
                return UnquantizedEmbeddingMethod()
            if is_linear:
                return UnquantizedLinearMethod()
            return None
        kind = self._index["tensors"][tensor_name]["kind"]
        if kind == "fp16_raw":
            if is_embedding:
                return UnquantizedEmbeddingMethod()
            if is_linear:
                return UnquantizedLinearMethod()
            return None
        if is_embedding:
            return LangBurstLowBitEmbeddingMethod(tensor_name=tensor_name, qb_model=self.qb_model)
        if is_linear:
            return LangBurstLowBitLinearMethod(tensor_name=tensor_name, qb_model=self.qb_model)
        return None

    def _resolve_tensor_name(self, prefix: str) -> str | None:
        tensors = self._index.get("tensors", {})
        for candidate in _candidate_weight_names(prefix):
            if candidate in tensors:
                return candidate
        return None

    def _audit_missing(self, layer: nn.Module, prefix: str) -> None:
        if not _env_enabled("LANGBURST_VLLM_LOWBIT_AUDIT"):
            return
        key = f"{type(layer).__name__}:{prefix}"
        if key in _AUDITED_MISSING:
            return
        _AUDITED_MISSING.add(key)
        print(f"[langburst_lowbit][missing] {key}", flush=True)
        if _env_enabled("LANGBURST_VLLM_LOWBIT_STRICT"):
            raise RuntimeError(f"LangBurst low-bit tensor not found for vLLM layer prefix: {prefix}")


try:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    if not issubclass(LangBurstLowBitConfig, QuantizationConfig):
        LangBurstLowBitConfig = type(
            "LangBurstLowBitConfig",
            (LangBurstLowBitConfig, QuantizationConfig),
            {},
        )
except Exception:
    pass


class LangBurstLowBitModelLoader:
    """vLLM loader that streams only fp16 LangBurst tensors.

    Quantized tensors are consumed lazily by ``LangBurstLowBit*Method`` so vLLM
    keeps ownership of the model graph and scheduler without materializing the
    27B checkpoint as dense fp16 weights.
    """

    def __init__(self, load_config: Any) -> None:
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        if not isinstance(self, BaseModelLoader):
            BaseModelLoader.__init__(self, load_config)
        self.load_config = load_config
        self.qb_model = _qb_model_from_config(getattr(load_config, "model_loader_extra_config", {}) or {})
        self.store = QuantizedStore(self.qb_model, device="cpu")

    def download_model(self, model_config: Any) -> None:
        return

    def load_weights(self, model: nn.Module, model_config: Any) -> None:
        self._repair_and_audit_quant_methods(model)
        model.load_weights(self._iter_fp16_weights())
        if _preload_enabled():
            device = torch.device("cuda")
            for param in model.parameters():
                if param.device.type == "cuda":
                    device = param.device
                    break
            _runtime_cache(self.qb_model, device, preload=True)
            options = _runtime_options()
            aux_devices = {
                aux_device
                for name in self.store.index.get("tensors", {})
                if (aux_device := options.aux_device_for_tensor(name)) is not None
            }
            for aux_device in aux_devices:
                _runtime_cache(self.qb_model, aux_device, preload=True)

    def _iter_fp16_weights(self) -> Iterable[tuple[str, torch.Tensor]]:
        for name, meta in self.store.index.get("tensors", {}).items():
            if meta.get("kind") != "fp16_raw":
                continue
            tensor = self.store.tensor(name)
            if isinstance(tensor, FP16Tensor):
                yield name, tensor.value

    def _repair_and_audit_quant_methods(self, model: nn.Module) -> None:
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        strict = _env_enabled("LANGBURST_VLLM_LOWBIT_STRICT")
        audit = _env_enabled("LANGBURST_VLLM_LOWBIT_AUDIT") or strict
        if not audit:
            return

        config = LangBurstLowBitConfig(self.qb_model)
        missing: list[str] = []
        repaired: list[str] = []
        unbound: list[str] = []
        for prefix, layer in model.named_modules():
            if not prefix:
                continue
            is_embedding = isinstance(layer, VocabParallelEmbedding)
            is_linear = isinstance(layer, (LinearBase, ParallelLMHead))
            if not (is_embedding or is_linear):
                continue
            tensor_name = config._resolve_tensor_name(prefix)
            if tensor_name is None:
                missing.append(f"{type(layer).__name__}:{prefix}")
                continue
            kind = config._index["tensors"][tensor_name]["kind"]
            if kind == "fp16_raw":
                continue
            quant_method = getattr(layer, "quant_method", None)
            if isinstance(quant_method, _LangBurstLowBitMixin):
                continue
            if is_embedding:
                layer.quant_method = LangBurstLowBitEmbeddingMethod(tensor_name=tensor_name, qb_model=self.qb_model)
                repaired.append(f"{type(layer).__name__}:{prefix}->{tensor_name}")
                continue
            if is_linear:
                layer.quant_method = LangBurstLowBitLinearMethod(tensor_name=tensor_name, qb_model=self.qb_model)
                repaired.append(f"{type(layer).__name__}:{prefix}->{tensor_name}")
                continue
            unbound.append(f"{type(layer).__name__}:{prefix}->{tensor_name}")

        for item in repaired[:200]:
            print(f"[langburst_lowbit][repair] {item}", flush=True)
        for item in missing[:200]:
            print(f"[langburst_lowbit][missing] {item}", flush=True)
        for item in unbound[:200]:
            print(f"[langburst_lowbit][unbound] {item}", flush=True)
        if strict and (missing or unbound):
            detail = "; ".join([*missing[:20], *unbound[:20]])
            raise RuntimeError(f"LangBurst low-bit strict audit failed for vLLM graph: {detail}")


try:
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if not issubclass(LangBurstLowBitModelLoader, BaseModelLoader):
        LangBurstLowBitModelLoader = type(
            "LangBurstLowBitModelLoader",
            (LangBurstLowBitModelLoader, BaseModelLoader),
            {},
        )
except Exception:
    pass

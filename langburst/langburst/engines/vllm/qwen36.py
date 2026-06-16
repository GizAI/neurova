from __future__ import annotations

from torch import nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
from vllm.model_executor.models.qwen3_5_mtp import (
    Qwen3_5MTP,
    Qwen3_5MultiTokenPredictor,
)
from vllm.model_executor.models.utils import PPMissingLayer, make_empty_intermediate_tensors_factory, maybe_prefix


class LangBurstQwen36MultiTokenPredictor(Qwen3_5MultiTokenPredictor):
    """Qwen3.5 MTP predictor with LangBurst low-bit embeddings.

    The target model intentionally stays on vLLM's native Qwen3.5 path.
    This shim is scoped to vLLM's optional MTP draft model because upstream
    Qwen3_5MTP creates its embedding without forwarding quant_config.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.compilation_config = vllm_config.compilation_config
        self.do_not_compile = True
        self.vocab_size = config.vocab_size
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        fc_quant = None if (quant_config and quant_config.get_name() == "modelopt_fp4") else quant_config
        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=fc_quant,
            prefix=f"{prefix}.fc",
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], model_config.hf_text_config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LangBurstQwen36MTP(Qwen3_5MTP):
    """vLLM Qwen3.5 MTP draft model with LangBurst low-bit checkpoint support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        nn.Module.__init__(self)
        self.config = config
        self.compilation_config = vllm_config.compilation_config
        self.do_not_compile = True
        self.model = LangBurstQwen36MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

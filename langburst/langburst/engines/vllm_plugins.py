from __future__ import annotations


def register() -> None:
    """Register LangBurst's vLLM extension points lazily.

    vLLM loads this function in every worker process through the
    ``vllm.general_plugins`` entry point. Keep imports inside the function so
    the native LangBurst engine remains usable without vLLM installed.
    """

    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.models import ModelRegistry

    from .vllm_lowbit import LangBurstLowBitConfig, LangBurstLowBitModelLoader

    register_quantization_config("langburst_lowbit")(LangBurstLowBitConfig)
    register_model_loader("langburst_lowbit")(LangBurstLowBitModelLoader)
    ModelRegistry.register_model(
        "Qwen3_5MTP",
        "langburst.engines.vllm_qwen36:LangBurstQwen36MTP",
    )

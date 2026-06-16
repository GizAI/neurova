"""Legacy LangBurst native engine implementation.

This package is the canonical owner for the in-process Qwen3.6/GDN runtime.
Top-level serving should use `langburst.engines` and select `--engine native`
instead of importing these modules directly.
"""

from .runtime import GenerationConfig, RuntimeEngine, sample_next, sample_next_tensor

__all__ = [
    "GenerationConfig",
    "RuntimeEngine",
    "sample_next",
    "sample_next_tensor",
]

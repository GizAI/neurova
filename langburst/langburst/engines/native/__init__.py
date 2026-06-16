"""LangBurst native engine package.

This package is the public native runtime facade and the default engine
provider. Implementation modules live beside the facade inside this package so
native remains a first-class runtime, not a compatibility shim.
"""

from .batch_worker import BatchGenerationHandle, BatchGenerationWorker
from .block_table import KVBlockTable
from .manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, load_model_specs
from .model_runner import BatchedModelRunner
from .provider import NativeBackend, NativeProvider
from .runtime import GenerationConfig, RuntimeEngine, sample_next, sample_next_tensor
from .scheduler import AdmissionController, ContinuousBatchScheduler

__all__ = [
    "AdmissionController",
    "BatchGenerationHandle",
    "BatchGenerationWorker",
    "BatchedModelRunner",
    "ContinuousBatchScheduler",
    "EngineManager",
    "EngineResourcePolicy",
    "GenerationConfig",
    "KVBlockTable",
    "ModelResourceSpec",
    "NativeBackend",
    "NativeProvider",
    "RuntimeEngine",
    "load_model_specs",
    "sample_next",
    "sample_next_tensor",
]

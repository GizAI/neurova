from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

from ..qwen36_impl.config import Qwen36_27B_TextConfig
from ...core.runtime import GenerationConfig, sample_next_tensor
from ...loader import LowBitMarlinTensor
from ..qwen36_impl.state import DecodeState


@dataclass(frozen=True)
class GraphSafetyReport:
    """Decode CUDA Graph readiness report.

    This is deliberately a gate, not a partial graph implementation.  CUDA
    Graph decode is only worth wiring into serving after the whole one-token
    decode step has static device buffers and no Python-managed position/KV
    mutation in the captured region.
    """

    graph_ready: bool
    greedy_argmax_device_safe: bool
    marlin_workspace_preallocated: bool
    device_position_counters: bool
    ring_kv_device_indexing: bool
    no_python_state_mutation: bool
    notes: tuple[str, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            note.removeprefix("BLOCKER: ")
            for note in self.notes
            if note.startswith("BLOCKER: ")
        )

    def as_text(self) -> str:
        rows = [
            f"graph_ready={str(self.graph_ready).lower()}",
            f"greedy_argmax_device_safe={str(self.greedy_argmax_device_safe).lower()}",
            f"marlin_workspace_preallocated={str(self.marlin_workspace_preallocated).lower()}",
            f"device_position_counters={str(self.device_position_counters).lower()}",
            f"ring_kv_device_indexing={str(self.ring_kv_device_indexing).lower()}",
            f"no_python_state_mutation={str(self.no_python_state_mutation).lower()}",
        ]
        if self.notes:
            rows.append("notes:")
            rows.extend(f"- {note}" for note in self.notes)
        return "\n".join(rows)


def _iter_marlin_tensors(obj: Any, *, seen: set[int] | None = None):
    seen = seen or set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if isinstance(obj, LowBitMarlinTensor):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_marlin_tensors(value, seen=seen)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _iter_marlin_tensors(value, seen=seen)
        return
    if hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            yield from _iter_marlin_tensors(value, seen=seen)


def marlin_workspace_preallocated(model: Any) -> bool:
    tensors = list(_iter_marlin_tensors(model))
    if not tensors:
        return True
    for tensor in tensors:
        if tensor.qweight.device.type != "cuda":
            return False
        if tensor._workspace is None:
            return False
        if not tensor._out_cache:
            return False
    return True


def inspect_decode1_graph_safety(
    *,
    model: Any | None,
    state: Any,
    gen_cfg: GenerationConfig | None = None,
) -> GraphSafetyReport:
    gen_cfg = gen_cfg or GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=0)
    notes: list[str] = []

    greedy_argmax_device_safe = gen_cfg.temperature <= 0 and gen_cfg.top_k == 0
    if greedy_argmax_device_safe:
        notes.append("OK: greedy sampling can use device argmax and defer CPU token readback.")
    else:
        notes.append("BLOCKER: graph decode requires greedy-only sampling first.")

    workspace_ready = marlin_workspace_preallocated(model) if model is not None else False
    if workspace_ready:
        notes.append("OK: Marlin output/workspace buffers are preallocated or no Marlin tensors are present.")
    else:
        notes.append("BLOCKER: warm all Marlin tensors so gemm_out reuses fixed output/workspace buffers.")

    pos = getattr(state, "pos", None)
    kv_len = getattr(state, "kv_len", None)
    device_position_counters = torch.is_tensor(pos) and torch.is_tensor(kv_len) and pos.device.type == "cuda" and kv_len.device.type == "cuda"
    if device_position_counters:
        notes.append("OK: pos/kv_len are CUDA tensors.")
    else:
        notes.append("BLOCKER: DecodeState.pos and kv_len are Python counters; graph capture needs fixed CUDA counter tensors.")

    ring_kv_device_indexing = False
    if isinstance(state, DecodeState):
        ring_kv_device_indexing = False
        if state.kv_window_policy != "ring":
            notes.append("BLOCKER: decode graph should use ring KV only; shift/error policies are correctness fallbacks.")
        else:
            notes.append("BLOCKER: ring KV still materializes logical views in Python for wrapped windows.")
    else:
        notes.append("BLOCKER: unknown state type cannot prove graph-safe KV indexing.")

    no_python_state_mutation = device_position_counters and ring_kv_device_indexing
    if not no_python_state_mutation:
        notes.append("BLOCKER: forward_one still mutates Python-visible state during decode.")

    graph_ready = all(
        (
            greedy_argmax_device_safe,
            workspace_ready,
            device_position_counters,
            ring_kv_device_indexing,
            no_python_state_mutation,
        )
    )
    return GraphSafetyReport(
        graph_ready=graph_ready,
        greedy_argmax_device_safe=greedy_argmax_device_safe,
        marlin_workspace_preallocated=workspace_ready,
        device_position_counters=device_position_counters,
        ring_kv_device_indexing=ring_kv_device_indexing,
        no_python_state_mutation=no_python_state_mutation,
        notes=tuple(notes),
    )


def verify_graph_safe_argmax() -> bool:
    if not torch.cuda.is_available():
        return True
    logits = torch.tensor([0.1, 2.0, 1.0], device="cuda", dtype=torch.float16)
    token = sample_next_tensor(logits, GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=0))
    return torch.is_tensor(token) and token.device.type == "cuda" and int(token.detach().cpu()) == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit langburst decode1 CUDA Graph readiness")
    parser.add_argument("--static", action="store_true", help="run a lightweight static audit without loading weights")
    args = parser.parse_args()

    if args.static:
        state = DecodeState.allocate(Qwen36_27B_TextConfig(), max_seq_len=8, device="cpu", kv_window_policy="ring")
        report = inspect_decode1_graph_safety(model=object(), state=state)
        print(report.as_text())
        return

    parser.error("only --static is currently supported; live model graph audit should run inside the benchmark harness")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FusionX D2 manifest compiler.

This compiler deliberately does not treat model-family names as native hardware
operators. High-level nodes are expanded into a small set of validated hardware
primitives. Unsupported nodes fail compilation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

PRIMITIVES = {
    "GEMM", "GEMV", "VECTOR", "RMSNORM", "SOFTMAX", "ROPE",
    "STATE_UPDATE", "MOE_TOPK", "MTP_VERIFY", "SPARSE_GATHER",
    "PATCHIFY3D", "ADALN", "DMA", "FENCE"
}

RECIPES: dict[str, list[str]] = {
    "DENSE_FFN": ["RMSNORM", "GEMM", "VECTOR", "GEMM"],
    "GQA_ATTENTION": ["RMSNORM", "GEMM", "ROPE", "GEMM", "SOFTMAX", "GEMM", "GEMM"],
    "GATED_ATTENTION": ["RMSNORM", "GEMM", "ROPE", "GEMM", "SOFTMAX", "GEMM", "VECTOR", "GEMM"],
    "GATED_DELTANET": ["RMSNORM", "GEMM", "VECTOR", "STATE_UPDATE", "GEMM"],
    "MLA_ATTENTION": ["RMSNORM", "GEMM", "ROPE", "GEMM", "SOFTMAX", "GEMM", "GEMM"],
    "SPARSE_ATTENTION": ["RMSNORM", "GEMM", "ROPE", "SPARSE_GATHER", "GEMM", "SOFTMAX", "GEMM", "GEMM"],
    "MOE_FFN": ["RMSNORM", "GEMM", "MOE_TOPK", "SPARSE_GATHER", "GEMM", "VECTOR", "GEMM"],
    "MTP_HEAD": ["RMSNORM", "GEMM", "MTP_VERIFY"],
    "VIDEO_TRANSFORMER": ["PATCHIFY3D", "ADALN", "GEMM", "ROPE", "GEMM", "SOFTMAX", "GEMM", "GEMM", "VECTOR", "GEMM"],
    "VIDEO_VAE": ["DMA", "GEMM", "VECTOR", "GEMM", "FENCE"],
    "VISION_ENCODER": ["PATCHIFY3D", "GEMM", "ROPE", "GEMM", "SOFTMAX", "GEMM", "DENSE_FFN"],
}


def expand_node(node: str) -> list[str]:
    if node in PRIMITIVES:
        return [node]
    if node not in RECIPES:
        raise ValueError(f"unsupported high-level op: {node}")
    out: list[str] = []
    for item in RECIPES[node]:
        out.extend(expand_node(item))
    return out


def compile_profile(profile: dict) -> dict:
    emitted: list[dict] = []
    for layer_index, node in enumerate(profile["graph"]):
        ops = expand_node(node)
        for op_index, primitive in enumerate(ops):
            if primitive not in PRIMITIVES:
                raise AssertionError(primitive)
            emitted.append({
                "layer": layer_index,
                "source_op": node,
                "primitive_index": op_index,
                "primitive": primitive,
            })
    return {
        "schema": "fusionx-d2-plan-v1",
        "model_id": profile["model_id"],
        "architecture": profile["architecture"],
        "module_count": profile.get("module_count", 1),
        "primitive_count": len(emitted),
        "plan": emitted,
        "unsupported": [],
        "claim_boundary": "operator legalization only; full checkpoint quality and silicon performance require separate qualification",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/compiled"))
    ns = ap.parse_args()
    profiles = ns.profiles or sorted(Path("compiler/profiles").glob("*.json"))
    ns.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in profiles:
        profile = json.loads(path.read_text())
        plan = compile_profile(profile)
        out = ns.out / f"{path.stem}.plan.json"
        out.write_text(json.dumps(plan, indent=2) + "\n")
        summary.append({"profile": path.name, "model_id": plan["model_id"], "primitives": plan["primitive_count"], "pass": True})
        print(f"PASS {plan['model_id']} -> {plan['primitive_count']} primitives")
    (ns.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

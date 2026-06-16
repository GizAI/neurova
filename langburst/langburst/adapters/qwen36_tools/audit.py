from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..qwen36_impl.config import Qwen36_27B_TextConfig
from ..qwen36_impl.model import WeightResolver


def gdn_layout_present(names: set[str], p: str) -> tuple[bool, str]:
    las = (f"{p}.linear_attn", f"{p}.linear_attention", f"{p}.self_attn")

    def any_name(cands):
        for name in WeightResolver.expand_candidates(tuple(cands)):
            if name in names:
                return True
        return False

    fused = any_name([f"{la}.in_proj_qkvz.weight" for la in las]) and any_name([f"{la}.in_proj_ba.weight" for la in las])
    split = all(any_name([f"{la}.{s}.weight" for la in las]) for s in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"))
    if fused:
        return True, "fused_qkvz_ba"
    if split:
        return True, "split_qkv_z_a_b"
    return False, "missing_gdn_projection_layout"


def required_names(cfg: Qwen36_27B_TextConfig, names: set[str]) -> list[tuple[str, tuple[str, ...]]]:
    def any_name(cands: tuple[str, ...]) -> bool:
        return any(name in names for name in WeightResolver.expand_candidates(cands))

    req: list[tuple[str, tuple[str, ...]]] = [
        ("embedding", ("model.embed_tokens.weight", "model.tok_embeddings.weight")),
        ("final_norm", ("model.norm.weight", "model.final_layernorm.weight")),
        ("lm_head", ("lm_head.weight", "model.lm_head.weight", "model.output.weight")),
    ]
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        req.extend([
            (f"layer_{i}_input_norm", (f"{p}.input_layernorm.weight", f"{p}.input_norm.weight")),
            (f"layer_{i}_post_norm", (f"{p}.post_attention_layernorm.weight", f"{p}.post_norm.weight")),
            (f"layer_{i}_mlp_down", (f"{p}.mlp.down_proj.weight",)),
        ])
        gate_up = (f"{p}.mlp.gate_up_proj.weight",)
        if any_name(gate_up):
            req.append((f"layer_{i}_mlp_gate_up", gate_up))
        else:
            req.extend([
                (f"layer_{i}_mlp_gate", (f"{p}.mlp.gate_proj.weight",)),
                (f"layer_{i}_mlp_up", (f"{p}.mlp.up_proj.weight",)),
            ])
        if cfg.layer_type(i) == "gdn":
            ok, layout = gdn_layout_present(names, p)
            if not ok:
                req.append((f"layer_{i}_gdn_projection_layout", tuple()))
            las = (f"{p}.linear_attn", f"{p}.linear_attention", f"{p}.self_attn")
            for label, suffixes in [
                ("out_proj", ("out_proj.weight",)),
                ("norm", ("norm.weight",)),
                ("conv", ("conv1d.weight", "conv.weight")),
                ("A_log", ("A_log",)),
                ("dt_bias", ("dt_bias",)),
            ]:
                req.append((f"layer_{i}_gdn_{label}", tuple(f"{la}.{s}" for la in las for s in suffixes)))
        else:
            sa = f"{p}.self_attn"
            qkv = (f"{sa}.qkv_proj.weight",)
            if any_name(qkv):
                req.append((f"layer_{i}_attn_qkv", qkv))
            else:
                req.extend([
                    (f"layer_{i}_attn_q", (f"{sa}.q_proj.weight",)),
                    (f"layer_{i}_attn_k", (f"{sa}.k_proj.weight",)),
                    (f"layer_{i}_attn_v", (f"{sa}.v_proj.weight",)),
                ])
            req.append((f"layer_{i}_attn_o", (f"{sa}.o_proj.weight",)))
    return req


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit converted LangBurst checkpoint key coverage")
    ap.add_argument("qb_model", type=Path)
    ap.add_argument("--hf-model", type=Path, default=None, help="HF model dir/config.json for shape override")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = Qwen36_27B_TextConfig.from_hf_config(args.hf_model) if args.hf_model and (args.hf_model / "config.json").exists() else Qwen36_27B_TextConfig()
    with open(args.qb_model / "langburst_index.json", "r", encoding="utf-8") as f:
        index = json.load(f)
    names = set(index["tensors"].keys())

    def present(candidates: tuple[str, ...]) -> bool:
        if not candidates:
            return False
        for name in WeightResolver.expand_candidates(candidates):
            if name in names:
                return True
        return False

    missing: list[dict] = []
    for label, candidates in required_names(cfg, names):
        if not present(candidates):
            missing.append({"label": label, "candidates": list(candidates)})

    gdn_layouts = {}
    for i in cfg.gdn_layers:
        ok, layout = gdn_layout_present(names, f"model.layers.{i}")
        gdn_layouts[str(i)] = layout if ok else "missing"
    quant_embed = any(n.endswith("embed_tokens.weight") and index["tensors"][n]["kind"] == "lowbit_symmetric_groupwise" for n in names)
    quant_gdn_split = any(n.endswith("in_proj_qkv.weight") and index["tensors"][n]["kind"] == "lowbit_symmetric_groupwise" for n in names)
    quant_bits = sorted({index["tensors"][n]["bits"] for n in names if index["tensors"][n]["kind"] == "lowbit_symmetric_groupwise"})
    mtp_like = sorted(n for n in names if ("mtp" in n.lower() or "nextn" in n.lower() or "next_n" in n.lower()))[:200]
    payload = {
        "format": index.get("format"),
        "tensor_count": len(names),
        "missing_count": len(missing),
        "missing": missing[:200],
        "quantized_embed": quant_embed,
        "quantized_gdn_split_projections": quant_gdn_split,
        "quant_bits": quant_bits,
        "gdn_layout_examples": dict(list(gdn_layouts.items())[:8]),
        "mtp_like_count": len(mtp_like),
        "mtp_like_examples": mtp_like[:30],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"format={payload['format']} tensors={payload['tensor_count']} "
            f"missing={payload['missing_count']} quant_embed={quant_embed} quant_gdn_split={quant_gdn_split} "
            f"quant_bits={quant_bits} mtp_like={payload['mtp_like_count']}"
        )
        print("GDN layout examples:", payload["gdn_layout_examples"])
        if missing:
            print("Missing required mappings, first 50:")
            for m in missing[:50]:
                print(f"- {m['label']}: {m['candidates']}")
        else:
            print("All required first-chat mappings are present.")
        if mtp_like:
            print("MTP-like tensors found; exact native-MTP head mapping still needs model-specific wiring:")
            for n in mtp_like[:30]:
                print(f"- {n}")


if __name__ == "__main__":
    main()

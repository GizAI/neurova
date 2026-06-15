from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Qwen36_27B_TextConfig
from .state import DecodeState
from .episodic_memory import EpisodicMemory
from .ttt_sidecar import TTTSidecarConfig, TTTSidecarMemory


def main() -> None:
    ap = argparse.ArgumentParser(description="QwenBurst infinite-streaming state demo without loading 27B weights")
    ap.add_argument("--out", type=Path, default=Path("/tmp/qwenburst_infinite_demo"))
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Qwen36_27B_TextConfig()
    state = DecodeState.allocate(cfg, max_seq_len=args.window, device=args.device, dtype=torch.float16, kv_window_policy="shift")
    # Mutate a tiny part of the state so fork/snapshot is visibly non-empty.
    first_layer = cfg.gdn_layers[0]
    state.gdn_states[first_layer][0, 0, 0] = 1.0
    state.gdn_conv_states[first_layer][0, 0] = 0.5
    state.pos = 42
    state.kv_len = min(42, args.window)
    snap = args.out / "demo.qbstate.pt"
    state.save_snapshot(snap, include_attention_kv=False)
    reloaded = DecodeState.load_snapshot(snap, cfg, device=args.device)

    mem = EpisodicMemory()
    mem.add_text("QwenBurst infinite mode keeps recent KV and compressed GDN state.", record_id="design")
    mem.add_text("TTT sidecar is updated during ingestion, not every decode token.", record_id="ttt")
    mem_path = args.out / "memory.json"
    mem.save(mem_path)
    hits = mem.search("compressed recurrent state", top_k=1)

    ttt = TTTSidecarMemory(TTTSidecarConfig(hidden_size=32, memory_rank=8), device=args.device)
    hidden = torch.randn(4, 32, dtype=torch.float16, device=args.device)
    ttt.update(hidden)
    read = ttt.read(hidden[-1])

    print(f"snapshot={snap}")
    print(f"snapshot_pos={reloaded.pos} gdn_mib={reloaded.gdn_state_bytes / 1024**2:.2f}")
    print(f"memory={mem_path} top_hit={hits[0].record_id} score={hits[0].score:.3f}")
    print(f"ttt_updates={ttt.updates} read_norm={read.float().norm().item():.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SKIP_PARTS = {
    ".cache",
    ".luma_cache",
    "__pycache__",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    full = str(path).lower()
    if "fineweb_edu" in full or "governed_fineweb_edu" in name:
        return "main_pretrain", "fineweb_edu"
    if "cosmopedia" in full:
        return "explanation_expansion", "cosmopedia"
    if "luma" in full:
        if "memory" in name or "slotproof" in name:
            return "verifiable_later", "luma_memory"
        if "chatml" in name or "sft" in name or "speak" in name:
            return "sft_later", "luma_sft"
        return "legacy_sources", "luma"
    if "mamba3" in full:
        if "curriculum" in name or "recall" in name or "programmatic" in name:
            return "verifiable_later", "mamba3_curriculum"
        return "legacy_sources", "mamba3"
    if "neurova" in name:
        if "chat" in name or "sft" in name or "speak" in name:
            return "sft_later", "neurova_sft"
        return "legacy_sources", "neurova"
    if "no_cheat" in name or "mcq" in name or "rlvr" in name:
        return "verifiable_later", "verified_skill"
    if "tinystories" in full or "stage_b_prose_mix" in full:
        return "legacy_sources", "legacy_speak_bootstrap"
    if path.suffix in {".jsonl", ".txt"}:
        return "legacy_sources", "raw_text"
    return "manifests", "metadata"


def safe_name(path: Path) -> str:
    parts = [p for p in path.parts if p not in {".", ""}]
    return "__".join(parts)


def link_once(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if mode == "none":
        return "indexed"
    if mode in {"hardlink", "auto"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    os.symlink(os.path.relpath(src, dst.parent), dst)
    return "symlink"


def iter_candidates(root: Path, out_dir: Path) -> list[Path]:
    exts = {".jsonl", ".json", ".txt"}
    out: list[Path] = []
    for base in [root / "neuromamba" / "data", root / "neuromamba" / "configs"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.is_relative_to(out_dir) and "sources" not in path.relative_to(out_dir).parts:
                continue
            if not path.is_relative_to(out_dir) and "corpus" in path.parts:
                continue
            if path.suffix.lower() in exts:
                out.append(path)
    return sorted(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the canonical Neurova corpus registry without duplicating data.")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("neuromamba/data/corpus"))
    p.add_argument("--link-mode", choices=["auto", "hardlink", "symlink", "none"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    by_hash: dict[str, dict] = {}
    for src in iter_candidates(root, out_dir):
        rel = src.relative_to(root)
        digest = sha256_file(src)
        category, family = classify(rel)
        duplicate_of = by_hash.get(digest, {}).get("canonical_relpath")
        if src.is_relative_to(out_dir):
            dst_rel = rel.relative_to(Path("neuromamba/data/corpus"))
            dst = src
            link_type = "in_place"
        else:
            dst_rel = Path(category) / family / safe_name(rel)
            dst = out_dir / dst_rel
            link_type = "duplicate"
        if duplicate_of is None:
            if link_type != "in_place":
                link_type = link_once(src, dst, args.link_mode)
            by_hash[digest] = {
                "canonical_relpath": str(Path("neuromamba/data/corpus") / dst_rel),
                "source_relpath": str(rel),
            }
        entries.append(
            {
                "source_relpath": str(rel),
                "canonical_relpath": None if duplicate_of else str(Path("neuromamba/data/corpus") / dst_rel),
                "duplicate_of": duplicate_of,
                "category": category,
                "family": family,
                "bytes": src.stat().st_size,
                "sha256": digest,
                "link_type": link_type,
            }
        )

    registry = {
        "name": "neurova-corpus-registry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "out_dir": str(out_dir.relative_to(root)),
        "policy": {
            "single_source_of_truth": "neuromamba/data/corpus/registry.json",
            "dedup_key": "file_sha256",
            "storage": "hardlink preferred, symlink fallback; duplicates are manifest-only",
        },
        "summary": {
            "files_seen": len(entries),
            "unique_files": sum(1 for e in entries if not e["duplicate_of"]),
            "duplicates": sum(1 for e in entries if e["duplicate_of"]),
            "bytes_seen": sum(e["bytes"] for e in entries),
            "unique_bytes": sum(e["bytes"] for e in entries if not e["duplicate_of"]),
        },
        "entries": entries,
    }
    (out_dir / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(registry["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

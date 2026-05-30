#!/usr/bin/env bash
# Neurova V6 — Quick start
# Usage: bash neurova.sh [mode]
#   mode: bf16 (default, fastest) | 4bit (low VRAM)

MODE="${1:-bf16}"
cd "$(dirname "$0")"
export V6_MODE="$MODE"
exec python3 neurova_v6.py

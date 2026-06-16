# Baseline protocol before claiming 100 tok/s

Run all competitors with identical prompts, sampling, context, and token budget.

## Workloads

1. short greedy coding: 512 prompt tokens, 2048 generation, temp=0
2. medium code edit: 4096 prompt tokens, 2048 generation, temp=0.2, top_p=0.95
3. long context: 32768 prompt tokens, 1024 generation, temp=0.2
4. creative: 1024 prompt tokens, 2048 generation, temp=0.8, top_p=0.95

## Metrics

- TTFT
- target forward/s
- emitted tok/s
- MTP proposed/accepted/emitted-per-target-step
- VRAM peak
- exact greedy parity against reference for first 256 generated tokens

## Pass condition

LangBurst can claim a win only if generated token sequence matches the reference
for greedy mode and emitted tok/s is higher at the same quantization quality.

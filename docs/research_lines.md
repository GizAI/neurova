# Neurova Research Lines

This is the single status file for active and archived model lines. Keep launch
defaults, training plans, and promotion claims aligned with this file.

## Current Canonical Structure

| Line | Status | Scope | Boundary |
| --- | --- | --- | --- |
| `saneflow` | Active research | From-scratch non-Transformer, non-Mamba causal LM focused first on natural sentence generation. | `neurova.sh saneflow ...`, `saneflow/`, `scripts/saneflow_*`, docs below. |
| `neurova_v6` | Legacy live CLI | Qwen-backed embedding memory assistant. | legacy `neurova.sh` modes. |
| `mamba3` | Reference / historical experiments | Prior recurrent-model experiments and benchmarks. | Explicit `neurova.sh mamba3 ...` only. |
| `luma` | Stopped archive | Slot-ledger memory prototype kept for reference, not promotion or default chat. | Explicit `NEUROVA_ALLOW_LUMA=1 neurova.sh luma ...` only. |

## Active Research Contract

The active from-scratch path is SaneFlow:

```text
No Transformer decoder blocks.
No Mamba kernels or Mamba clone dependency.
No slot dependency for the first speaking baseline.
Natural continuation first; ChatML/SFT only after the base model speaks.
```

The first target is coherent English generation from a small independent
architecture:

```text
16K byte-level BPE
local causal depthwise convolution
gated multi-timescale recurrent state
SwiGLU
tied LM head
```

Detailed active-line notes live in:

- `docs/saneflow_from_scratch_plan.md`
- `docs/saneflow_research_actions.md`
- `docs/neurova_r_research_report.md`
- `docs/neurova_training_data_master_plan.md`

Machine-readable curriculum policy:

- `configs/saneflow_training_curriculum.json`

## Promotion Gate

A checkpoint is not promoted as a speaking baseline until fixed prompts produce
readable, non-collapsed answers:

```text
Hi. Who are you?
Tell me a short story about a robot and a garden.
Explain what a computer is in simple English.
What is the capital of France?
Write one sentence about the moon.
```

Loss, low perplexity, or narrow curriculum success is insufficient.

## LUMA Boundary

LUMA is stopped because the project goal changed to a clean, independent
speaking model. Its files remain for provenance and ideas, but they must not be
used as the default launcher, current checkpoint target, or promotion path.

Any future LUMA revival needs a separate decision and must not be mixed into the
SaneFlow training plan.

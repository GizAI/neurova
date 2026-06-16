# NeuroMamba

This directory is the NeuroMamba experimental path for Neurova.

Scope:
- official `state-spaces/mamba` implementation only
- MIMO-R4 target path, with `mimo-r4-tiny` for 16GB kernel validation
- Llama-3.1 tokenizer for the English-first path
- UTF-8 byte tokenizer only for dependency-free smoke tests
- recurrent state checkpoint proof via Mamba-3 `InferenceParams`
- paper-matched MIMO-R4 presets for 180M, 440M, 880M, and 1.5B targets
- RTX 4080 fast decode fix: Mamba-3 step `tile_D` follows `headdim`, not a hard-coded 64

It is not a finished 1.5B Korean model. It is the runnable development base for:
1. tiny smoke training
2. Mamba-3 SISO/MIMO kernel validation
3. recurrent state save/restore contract validation
4. later continued pretraining and instruction tuning

## Local Commands

```bash
./neurova.sh mamba3 "The main idea is"
./neurova.sh mamba3 eval
./neurova.sh mamba3 bench "The main idea is"
./neurova.sh mamba3 diagnose "The main idea is"
./neurova.sh mamba3 probe
./neurova.sh mamba3 serve
neuromamba/scripts/mamba3_run_gates.sh

python -m neuromamba.cli model-info --mode mimo-r4-1.5b --tokenizer llama31
python -m neuromamba.cli check-contract --mode mimo-r4-1.5b --tokenizer llama31
python -m neuromamba.cli smoke --mode mimo-r4-tiny --tokenizer llama31 --seq-len 256 --device cuda
python -m neuromamba.cli train-tiny --mode mimo-r4-tiny --tokenizer llama31 --seq-len 256 --steps 20 --device cuda --data luma/data/english_bootstrap.txt
python -m neuromamba.cli diagnose-decode --mode mimo-r4-tiny --tokenizer llama31 --seq-len 128 --device cuda
python -m neuromamba.cli bench-decode --mode mimo-r4-tiny --tokenizer llama31 --seq-len 128 --device cuda --cuda-graph --max-new 64
python -m neuromamba.cli state-prefill --mode mimo-r4-tiny --tokenizer llama31 --seq-len 256 --device cuda
python -m neuromamba.cli verify-rlvr --rlvr-data luma/data/rlvr_verifier_bootstrap.jsonl
```

## ml-dmc8

```bash
bash neuromamba/scripts/deploy_mamba3_ml_dmc8.sh
```

Defaults:
- env: `mamba3_siso`
- mode: `mimo-r4-tiny`
- tokenizer: `llama31`
- seq length: `512`
- training steps: `100`

If Hugging Face blocks the gated Meta tokenizer, accept the Llama-3.1 license and set `HF_TOKEN` or run `huggingface-cli login` on `ml-dmc8`.

```bash
TOKENIZER=llama31 HF_TOKEN=... bash neuromamba/scripts/deploy_mamba3_ml_dmc8.sh
```

MIMO needs the official Mamba-3 TileLang path to build correctly.

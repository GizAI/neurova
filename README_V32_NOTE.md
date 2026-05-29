# BrainOS V32 note

Runtime remains no-autoregressive and no-next-token. V32 adds predictive developmental learning infrastructure, not a GPT-style language model.

Official benchmark claim policy:

- `brainos/official_benchmark_loaders.py` evaluates user-supplied official SCAN/bAbI/CLUTRR-like files.
- If files are absent, it reports `loaded=false` and does not claim a score.
- Generated compatible audits are labeled as compatible generated subsets only.

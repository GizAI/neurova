# No-LLM Design Contract

V17 rejects the tiny-LM path. It must not implement autoregressive next-token or next-byte generation as the language core.

Allowed learning targets:

- IR type classification
- slot extraction
- phrase-to-IR fragment composition
- candidate ranking
- evidence relevance scoring
- proof operator selection
- memory action selection
- world transition plausibility
- contradiction/refutation scoring

Forbidden as core language ability:

- `generate()` text continuation as cognition
- next-token loss as proof of intelligence
- smoke-test success as open-domain language ability

Runtime cognition path:

```text
text → IR candidates → verifier → evidence graph → reasoner/executor/world model → rendered answer
```

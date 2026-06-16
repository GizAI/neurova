# Codex + Qwen EXL3 Runbook

Last verified: 2026-05-31 KST on `ml-dmc8`.

## Current Live State

- Codex default model: `neurova/qwen` via `cliproxyapi` in `~/.codex/config.toml`.
- CLI proxy endpoint: `http://127.0.0.1:8317/v1`, `wire_api = "responses"`.
- Proxy model alias: `neurova/qwen` -> `http://ml-dmc8:5000/v1` -> `Qwen3.6-27B-exl3-3.08bpw`.
- TabbyAPI service: `neurova-tabbyapi.service`, enabled and active.
- WGP services on `ml-dmc8`: `wgp-api1`, `wgp-api2` disabled/inactive, not masked.

Do not kill or restart the local `cli-proxy-api` while Codex is running through it.

## Stable TabbyAPI Config

`/home/user/tabby-models/tabbyAPI/config.yml`:

```yaml
model:
  model_name: Qwen3.6-27B-exl3-3.08bpw
  max_seq_len: 100352
  cache_size: 100352
  cache_mode: "3,2"
  chunk_size: 256
  output_chunking: true
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  tool_format: qwen3_coder
  max_batch_size: 2

draft_model:
  draft_model_name: Qwen3.6-27B-DFlash-exl3
  draft_cache_mode: "Q4"
  draft_num_tokens: 6
```

Why this setting:

- `100352 + max_batch_size 2 + draft_num_tokens 6` is the stable coding profile on RTX 4080 16GB.
- `65536 + draft_num_tokens 8` is faster for short turns, but real Codex coding can exceed 65K context. A 71K prompt failed at 65K and looked like a dead model.
- `draft_num_tokens 8+` at 100K/batch 2 leaves too little VRAM and can fail with `Insufficient VRAM in split for model and cache`.
- Short turns show roughly `35-58 T/s` generation. A 71K Codex context completed, but took about 145 seconds because prefill became the bottleneck.

## Required Patches

These are captured in `deploy_exl3/setup_tabbyapi_exl3.sh` so the setup can be replayed.

1. Model config compatibility:
   - File: `/home/user/tabby-models/models/Qwen3.6-27B-exl3-3.08bpw/config.json`
   - Set `architectures` to `Qwen3_5ForCausalLM`.
   - Copy `text_config` fields to the root object.

2. Chat template no-think default and final language guard:
   - Files: `chat_template.jinja`, `tokenizer_config.json`
   - Local source of truth: `deploy_exl3/templates/qwen3_coder_neurova_chat_template.jinja`.
   - Default `enable_thinking` to false unless explicitly enabled.
   - Add short final-answer guard: `Final answer must be in Korean unless the user asks otherwise. Do not include Chinese text in final answer. Never end with a promise of future action; if work remains, call tools in the same reply, otherwise provide completed results.`
   - The tool instruction block also says that if the model says it will run/check/edit/verify/retry/execute, it must include the matching tool call in the same reply.
   - This prevents Codex/Qwen from ending a turn with only status text like `이제 cleanly 다시 시작합니다.` or `이제 다시 실행한다.`.

3. EXL3 byte-fallback streaming decode fix:
   - File: `/home/user/tabby-models/tabbyAPI/backends/exllamav3/model.py`
   - Do not trust `result["text"]` for streamed chunks.
   - Accumulate generated token ids, decode the full generated id list, and only emit stable text before any `U+FFFD` suffix.
   - This fixes Korean corruption such as `토큰` becoming replacement characters in direct TabbyAPI, proxy, and Codex paths.

4. WGP cleanup:
   - `wgp-api1` and `wgp-api2` are disabled/inactive on `ml-dmc8`.
   - They are not masked, so they can still be manually started later.
   - Monitoring config should not auto-restart WGP on `ml-dmc8`.

## Verification Commands

Check live service and config:

```bash
ssh ml-dmc8 'systemctl is-enabled neurova-tabbyapi; systemctl is-active neurova-tabbyapi; grep -n "max_seq_len\|cache_size\|max_batch_size\|draft_model_name\|draft_num_tokens\|tool_format" /home/user/tabby-models/tabbyAPI/config.yml'
```

Force Codex to use Qwen even if local config drifts:

```bash
timeout 180 codex exec --ephemeral --json --sandbox read-only \
  -C /home/user/workspace/neurova \
  -c approval_policy='"never"' \
  -c model='"neurova/qwen"' \
  -c model_provider='"cliproxyapi"' \
  -c model_reasoning_effort='"low"' \
  '도구를 사용해서 README.md, neurova/v6.py, deploy_exl3/README_TABBYAPI_EXL3.md를 확인해. 한국어로 8줄 이내로 요약해. 파일은 수정하지 마.'
```

Watch TabbyAPI metrics while Codex runs:

```bash
ssh ml-dmc8 'tail -f /home/user/tabby-models/tabbyAPI/logs/$(ls -t /home/user/tabby-models/tabbyAPI/logs | head -n1)'
```

Expected explicit-Qwen complex read-only result from 2026-05-31:

- First request: `Context: 16089`, `Generate: 58.44 T/s`, 3 parsed tool calls, about 20 seconds.
- Follow-up request: `Context: 22254`, `Generate: 46.15 T/s`, about 15 seconds.
- 71K recovery request after restoring 100K context: `Context: 71017`, completed in about 145 seconds.
- Output JSONL: `fffd 0`, `has_efbfbd False`.
- Progress-only guard verification after Tabby restart: `guard_present True`; explicit Codex read-only test called `/bin/bash -lc 'head -30 deploy_exl3/CODEX_QWEN_RUNBOOK.md ...'` and completed normally.
- Follow-up guard verification after Tabby restart: `same_reply_guard True`, `future_action_guard True`; explicit Codex test executed both `ls deploy_exl3/templates` and `pwd`, then returned final results instead of ending with `다시 실행한다`.

## Practical Limit

This is usable for real Codex coding when prompts are kept bounded and `model_reasoning_effort=low` is used for routine edits. The bottleneck is not DFlash generation; it is Codex sending large Responses API contexts and tool schemas every turn. For very large repos, prefer focused prompts, fewer requested files, and explicit `-c model_reasoning_effort='"low"'` unless deep reasoning is required.

# CodexProxy / CLIProxyAPI Integration

This directory is the local source of truth for running Codex through
CLIProxyAPI against LangBurst.

It intentionally stores only reproducible integration assets:

- `codexproxy`: small launcher that keeps plain `codex` unchanged and uses the
  proxy only when invoked as `codexproxy`.
- `proxy.config.toml`: Codex profile template for the Responses API wire.
- `cli-proxy-api.config.yaml`: secret-free CLIProxyAPI config template for
  LangBurst.
- `patches/cliproxyapi-openai-responses-developer-role.patch`: required
  upstream patch for CLIProxyAPI.

Do not store API keys, tokens, generated binaries, or machine-specific model
state here.

## Why the CLIProxyAPI patch is required

Codex sends OpenAI Responses input with a leading `developer` message. The
unpatched CLIProxyAPI conversion changed that role to `user` when bridging
Responses to Chat Completions. For LangBurst/Qwen this made Codex's internal
developer instructions and repository context look like ordinary user text, so
Qwen answered the agent prelude instead of the latest user request.

The patch changes:

```text
Responses developer role -> Chat Completions system role
```

and adds a regression test.

LangBurst additionally treats `tools + tool_choice:auto` on the native provider
as a plain-text agent request: it renders the latest user task without feeding
the entire Codex agent prelude to Qwen. Forced `tool_choice` still requires a
tool-capable engine.

## Install / refresh

From this repository:

```bash
./langburst/integrations/codexproxy/install.sh
```

The installer backs up existing `~/.codex/proxy.config.toml` and
`~/.cli-proxy-api/config.yaml` before writing the LangBurst integration files.
If the machine has a mixed CLIProxyAPI config for other providers, merge from
the backup instead of treating this template as the only desired config.

Optional environment overrides:

```bash
CLIPROXYAPI_SRC=/home/user/opensources/CLIProxyAPI \
LANGBURST_BASE_URL=http://192.168.0.47:8008/v1 \
./langburst/integrations/codexproxy/install.sh
```

## Smoke tests

```bash
curl -fsS -H 'Authorization: Bearer sk-codex-local' \
  http://127.0.0.1:8317/v1/models

codexproxy -m neurova/qwen exec \
  --output-last-message /tmp/codex_qwen_last.txt \
  "가계부앱 만들어줘. 첫 문장은 한국어로 해."

sed -n '1,40p' /tmp/codex_qwen_last.txt
```

Expected behavior: the answer starts with the budget-app task, not with Codex
developer/system prompt analysis.

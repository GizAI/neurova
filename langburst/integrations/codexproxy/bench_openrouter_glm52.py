#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request


DEFAULT_ENV_FILE = pathlib.Path(__file__).resolve().parents[4] / "giz/giz.env"
MODEL = "z-ai/glm-5.2"
PROVIDERS = ("together", "friendli", "akashml")


def load_env_file(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def request_once(provider: str) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    url = "https://openrouter.ai/api/v1/chat/completions"
    prompt = (
        "한국어로 140~180 토큰 정도로 답해줘. "
        "주제는 OpenRouter provider routing 성능 비교다. "
        "한 문단과 세 개의 짧은 불릿을 포함해줘."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "max_tokens": 160,
        "temperature": 0.2,
        "stream_options": {"include_usage": True},
        "provider": {"only": [provider]},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://www.giz.ai",
            "X-Title": "LangBurst",
            "Accept": "text/event-stream",
        },
    )

    t0 = time.perf_counter()
    first_token_ts = None
    completion_tokens = None
    finish_reason = None
    text_bytes = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith(b"data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == b"[DONE]":
                    break
                if first_token_ts is None:
                    first_token_ts = time.perf_counter()
                try:
                    obj = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage") and isinstance(obj["usage"], dict):
                    completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
                choices = obj.get("choices") or []
                if choices:
                    choice = choices[0] or {}
                    finish_reason = choice.get("finish_reason", finish_reason)
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if isinstance(text, str):
                        text_bytes += len(text.encode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{provider}: HTTP {e.code}: {detail}") from e

    t1 = time.perf_counter()
    ttft = (first_token_ts - t0) if first_token_ts is not None else None
    e2e = t1 - t0
    decode_window = (e2e - ttft) if ttft is not None else None
    decode_tok_s = None
    if isinstance(completion_tokens, int) and decode_window and decode_window > 0:
        decode_tok_s = completion_tokens / decode_window

    return {
        "provider": provider,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "completion_tokens": completion_tokens,
        "decode_tok_s": decode_tok_s,
        "finish_reason": finish_reason,
        "text_bytes": text_bytes,
    }


def main() -> int:
    env_file = pathlib.Path(os.environ.get("OPENROUTER_ENV_FILE", str(DEFAULT_ENV_FILE))).expanduser()
    load_env_file(env_file)

    print(f"model={MODEL}")
    print(f"env_file={env_file}")
    rows = []
    for provider in PROVIDERS:
        row = request_once(provider)
        rows.append(row)
        ttft = row["ttft_s"]
        e2e = row["e2e_s"]
        tok_s = row["decode_tok_s"]
        comp = row["completion_tokens"]
        print(
            f'{provider}: ttft={ttft:.3f}s e2e={e2e:.3f}s '
            f'completion_tokens={comp} decode_tok_s={tok_s:.2f} finish={row["finish_reason"]}'
        )

    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for the persistent Neurova chat server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = "/stream" if args.stream else "/generate"
    url = f"http://{args.host}:{args.port}{path}"
    body = json.dumps({"prompt": args.prompt, "max_new": args.max_new}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=None) as resp:
        if args.stream:
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
            return
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload.get("ok", False):
        raise SystemExit(payload.get("error", "generation failed"))
    print(payload["answer"])
    print(f"({payload['tok_s']:.1f} tok/s)")


if __name__ == "__main__":
    main()

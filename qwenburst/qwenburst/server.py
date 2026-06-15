from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from .config import Qwen36_27B_TextConfig
from .dflash import DFlashDraftAdapter
from .generate import GenerationConfig, choose_weight_device, load_tokenizer, sample_next
from .loader import QuantizedStore
from .model import QwenBurstModel
from .state import DecodeState


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str = "qwenburst-qwen3.6-27b-q3"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_k: int = 0
    stream: bool = False


@dataclass
class QwenBurstEngine:
    hf_model: Path
    qb_model: Path
    device: str
    recent_window: int
    weight_device: str
    gpu_embed_head: bool
    model_name: str
    draft_adapter: DFlashDraftAdapter | None = None

    def __post_init__(self) -> None:
        self.cfg = Qwen36_27B_TextConfig.from_hf_config(self.hf_model)
        self.tokenizer = load_tokenizer(self.hf_model)
        resolved_weight_device = choose_weight_device(self.qb_model, self.weight_device, self.device)
        store = QuantizedStore(self.qb_model, device=resolved_weight_device)
        embed_store = None if self.gpu_embed_head else QuantizedStore(self.qb_model, device="cpu")
        head_store = None if self.gpu_embed_head else QuantizedStore(self.qb_model, device="cpu")
        self.model = QwenBurstModel(store, cfg=self.cfg, device=self.device, embed_store=embed_store, head_store=head_store)
        self.lock = threading.Lock()

    def encode_messages(self, messages: list[ChatMessage]) -> list[int]:
        payload = [{"role": m.role, "content": self._content_to_text(m.content)} for m in messages]
        encoded = self.tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.reshape(-1).tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(t) for t in encoded]

    @staticmethod
    def _content_to_text(content: str | list[dict[str, Any]] | None) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)

    def completion_tokens(self, req: ChatCompletionRequest):
        if self.draft_adapter is not None:
            raise RuntimeError(
                "DFlash draft adapter is loaded, but qwenburst-native DFlash proposal execution "
                "is not implemented yet. The target runtime was not replaced or bypassed."
            )
        max_new = req.max_new_tokens or req.max_completion_tokens or req.max_tokens or 256
        prompt_ids = self.encode_messages(req.messages)
        state = DecodeState.allocate(self.cfg, max_seq_len=self.recent_window, device=self.device, kv_window_policy="ring")
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new,
            temperature=req.temperature,
            top_k=req.top_k,
            eos_token_ids=self.eos_token_ids(),
        )
        with self.lock, torch.no_grad():
            logits: torch.Tensor | None = None
            for i, tid in enumerate(prompt_ids):
                logits = self.model.forward_one(tid, state, return_logits=(i == len(prompt_ids) - 1))
            assert logits is not None
            next_id = sample_next(logits, gen_cfg)
            for _ in range(max_new):
                if gen_cfg.eos_token_ids and next_id in gen_cfg.eos_token_ids:
                    break
                text = self.tokenizer.decode([next_id], skip_special_tokens=False)
                yield next_id, text
                logits = self.model.forward_one(next_id, state, return_logits=True)
                next_id = sample_next(logits, gen_cfg)

    def eos_token_ids(self) -> tuple[int, ...]:
        ids = []
        for name in ("eos_token_id", "pad_token_id"):
            val = getattr(self.tokenizer, name, None)
            if isinstance(val, int):
                ids.append(val)
        return tuple(set(ids))


def create_app(engine: QwenBurstEngine):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qwenburst-server requires fastapi and uvicorn") from exc

    app = FastAPI(title="QwenBurst OpenAI-Compatible Server")

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "qwenburst",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if req.stream:
            def events():
                for _, text in engine.completion_tokens(req):
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": engine.model_name,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield "data: " + json.dumps(done, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        pieces = [text for _, text in engine.completion_tokens(req)]
        content = "".join(pieces)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": engine.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible QwenBurst thin server")
    ap.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    ap.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb3"))
    ap.add_argument("--model-name", default="qwenburst-qwen3.6-27b-q3")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=8192)
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--gpu-embed-head", action="store_true")
    ap.add_argument("--speculative-backend", choices=("none", "dflash"), default="none")
    ap.add_argument("--draft-model", default="z-lab/Qwen3.6-27B-DFlash")
    ap.add_argument("--dflash-draft-dir", type=Path, default=None, help="converted qwenburst low-bit DFlash draft directory")
    ap.add_argument("--num-speculative-tokens", type=int, default=15)
    args = ap.parse_args()

    draft_adapter = None
    if args.speculative_backend == "dflash":
        if args.dflash_draft_dir is None:
            raise SystemExit(
                "speculative-backend=dflash requires --dflash-draft-dir pointing to a converted "
                "qwenburst low-bit DFlash draft. Convert with: python -m qwenburst.dflash convert <hf_dflash_dir> <out_dir> --bits 3"
            )
        draft_adapter = DFlashDraftAdapter.from_lowbit_dir(args.dflash_draft_dir, device=args.device)
    elif args.speculative_backend != "none":
        raise SystemExit(
            "unknown speculative backend. qwenburst only supports target-local adapter paths."
        )

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qwenburst-server requires uvicorn") from exc

    engine = QwenBurstEngine(
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        gpu_embed_head=args.gpu_embed_head,
        model_name=args.model_name,
        draft_adapter=draft_adapter,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

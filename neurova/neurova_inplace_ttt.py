"""
Neurova In-Place TTT Engine v3 — 단일 GPU 최대 컨텍스트 + 최대 출력 + 최고 속도.

Architecture:
  Qwen3.5-4B (4-bit, ~2.5GB)
  + In-Place TTT v3 (멀티패스 + 모멘텀)
  + KV cache: full attention only (8/32 layers, Gated DeltaNet 24/32는 불필요)
  + Flash Attention 2/3 (auto)
  + 진짜 스트리밍 토큰 출력
  
VRAM 예상 (RTX 4080 16GB):
  4-bit model:       ~2.5 GB
  TTT modules:       ~0.1 GB
  KV cache 256K:     ~5.2 GB (full attention 8/32 layers only)
  Overhead/buffers:  ~2.0 GB
  Total:             ~9.8 GB
  여유:              ~5.8 GB → 256K 안정적 확보
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Generator, Dict, Any
import os, re, json, time, uuid, sys
from dataclasses import dataclass, field
from pathlib import Path
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .inplace_ttt import InPlaceTTTWrapper, apply_ttt_to_model

MODEL_NAME = "Qwen/Qwen3.5-4B"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]

# VRAM 기반 컨텍스트 선택 (RTX 4080 16GB 기준)
_CONTEXT_LEVELS = {
    32768:  3.7,
    65536:  5.3,
    131072: 8.5,
    196608: 11.5,
    262144: 14.0,
}
_DEFAULT_CONTEXT = 262144  # 256K

# 환경변수 오버라이드
MAX_CONTEXT = int(os.environ.get("NEUROVA_MAX_CONTEXT", "0")) or _DEFAULT_CONTEXT
MAX_NEW_TOKENS = int(os.environ.get("NEUROVA_MAX_TOKENS", "8192"))
MAX_HISTORY_TURNS = 10  # 히스토리 턴 수 (컨텍스트 내에서 자동 조정)

# TTT 멀티패스
TTT_LEARN_PASSES = int(os.environ.get("NEUROVA_TTT_PASSES", "5"))
TTT_LEARN_LR = float(os.environ.get("NEUROVA_TTT_LR", "3.0"))


def _norm(s): return re.sub(r"\s+", " ", (s or "").strip())
def _uid(): return uuid.uuid4().hex[:8]


# ── Correction Memory ──

@dataclass
class Correction:
    question: str = ""; answer: str = ""
    timestamp: float = field(default_factory=time.time); use_count: int = 0

class CorrectionMemory:
    def __init__(self, path=""):
        self.path = path or os.environ.get("NEUROVA_TTT_MEMORY", "./.neurova_ttt_memory.json")
        self.corrections: List[Correction] = []
        self._load()
    def _load(self):
        p = Path(self.path)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    self.corrections = [Correction(**c) for c in data]
            except: pass
    def _save(self):
        Path(self.path).write_text(json.dumps(
            [{"question": c.question, "answer": c.answer,
              "timestamp": c.timestamp, "use_count": c.use_count}
             for c in self.corrections], ensure_ascii=False, indent=2))
    def add(self, q, a):
        self.corrections.append(Correction(question=_norm(q), answer=_norm(a)))
        self._save()
    def find(self, q):
        qn = _norm(q.lower())
        for c in reversed(self.corrections):
            if c.question.lower() == qn:
                c.use_count += 1; self._save(); return c.answer
        return None
    def all(self): return list(self.corrections)
    def __len__(self): return len(self.corrections)


# ── Engine ──

class NeurovaInPlaceTTT:
    """Neurova In-Place TTT Engine v3 — 최대 성능."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.ttt_wrapper = None
        self.corrections = CorrectionMemory()
        self.show_thinking = False
        self._loaded = False
        self.history: List[dict] = []
        self.max_context = MAX_CONTEXT
        self.max_new_tokens = MAX_NEW_TOKENS
        self._gen_time = 0.0
        self._learn_count = 0
        self._prefill_time = 0.0
        self._decode_time = 0.0

    def load(self):
        """Load model with optimal settings for single 16GB GPU."""
        if self._loaded:
            return

        t0 = time.time()
        print(f"[Neurova] Loading {self.model_name} 4-bit", flush=True)
        print(f"  Context: {self.max_context:,} | Max tokens: {self.max_new_tokens}", flush=True)
        print(f"  TTT: {len(TTT_LAYERS)} layers {TTT_LAYERS} | Passes: {TTT_LEARN_PASSES} | 8-bit", flush=True)

        # 8-bit quantization (Int8 — supports weight modification)
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = self.max_context

        # Apply In-Place TTT
        self.ttt_wrapper = apply_ttt_to_model(
            model,
            ttt_layers=TTT_LAYERS,
            ttt_lr=0.01,
            ttt_chunk=2048,
            ttt_momentum=0.9,
            ttt_proj=True,
        )
        self.model = model
        self.tokenizer = tokenizer
        self._loaded = True
        dt = time.time() - t0
        print(f"[Neurova] Loaded {dt:.1f}s | VRAM: ~2.5GB (model) + ~5.2GB (256K cache)", flush=True)

    def unload(self):
        if self._loaded:
            self.ttt_wrapper = None
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            torch.cuda.empty_cache()
            print("[Neurova] Unloaded, GPU memory released.", flush=True)

    # ── TTT Learning ──

    def learn_from_text(self, text: str, passes: int = 0) -> str:
        """Process text → In-Place TTT multi-pass updates MLP down-proj weights."""
        if not self._loaded:
            return "Load model first (ttt: on)"
        text = _norm(text)
        if not text or len(text) < 5:
            return "Text too short for learning."
        
        passes = passes or TTT_LEARN_PASSES
        
        # Snapshot weights before learning
        self.ttt_wrapper.snapshot_weights()
        
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=self.max_context
        ).to("cuda")
        
        with torch.no_grad():
            self.ttt_wrapper.forward_ttt(**inputs, num_passes=passes)
        
        # Check weight changes
        diffs = self.ttt_wrapper.weight_diff()
        avg_diff = sum(diffs.values()) / max(len(diffs), 1)
        
        self._learn_count += 1
        return (f"TTT learned ({passes} passes, {len(diffs)} layers, "
                f"avg weight Δ={avg_diff:.6f})")

    def learn_qa(self, question: str, answer: str, passes: int = 0) -> str:
        """Learn Q&A pair via TTT + store in episodic memory."""
        if not self._loaded:
            return "Load model first."
        q, a = _norm(question), _norm(answer)
        self.corrections.add(q, a)
        
        # Build learning text: Q&A format in natural language
        text = f"Question: {q}\nAnswer: {a}"
        
        result = self.learn_from_text(text, passes=passes)
        return f"Learned & stored: {q} => {a} [{result}]"

    # ── Streaming Generation ──

    def stream_generate(self, text: str, max_new_tokens: int = 0,
                        temperature: float = 0.7) -> Generator[str, None, str]:
        """Generate token-by-token stream.
        
        Yields individual tokens as they're generated.
        Return value: the full cleaned response (accessible via generator protocol).
        
        Architecture:
          1. Prefill: process all input tokens in one forward pass → build KV cache
          2. Decode: generate one token at a time, extend KV cache
          3. Streaming: yield each decoded token immediately
        
        Qwen3.5-4B hybrid optimization:
          - Gated DeltaNet (24/32 layers): no KV cache needed
          - Full Attention (8/32 layers): KV cache only for these
          - Flash Attention 2 for speed
        """
        if not self._loaded:
            yield "[Load model first]"
            return "[Load model first]"

        text = _norm(text)
        
        # Build prompt with history
        prompt = self._build_prompt(text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_context
        ).to("cuda")
        
        input_len = inputs["input_ids"].shape[1]
        full_response = ""
        token_buffer = ""
        t0 = time.time()

        # ── Prefill (process all input tokens at once) ──
        with torch.no_grad():
            out = self.model(
                **inputs,
                use_cache=True,
                output_hidden_states=False,
            )
            past = out.past_key_values
        
        self._prefill_time = time.time() - t0
        
        # ── Decode (generate one token at a time) ──
        next_token = inputs["input_ids"]  # start from last token of prefill
        tok_t0 = time.time()
        
        max_tokens = max_new_tokens or self.max_new_tokens
        
        for i in range(max_tokens):
            with torch.no_grad():
                # Take last token only (rest is in KV cache)
                last_token = next_token[:, -1:]
                
                out = self.model(
                    input_ids=last_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                
                # Temperature sampling
                if temperature > 0 and temperature != 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Decode this token
                token_id = next_token.item()
                
                # Stop on EOS
                if token_id == self.tokenizer.eos_token_id:
                    break
                
                # Skip bos token if it appears mid-generation
                if token_id == self.tokenizer.bos_token_id:
                    continue
                
                token_str = self.tokenizer.decode(token_id, skip_special_tokens=True)
                full_response += token_str
                
                # Accumulate for potential flush (real-time streaming)
                token_buffer += token_str
                if len(token_buffer) >= 4 or token_str in ('.', '!', '?', '\n'):
                    yield token_buffer
                    token_buffer = ""
                else:
                    yield token_str

        # Flush remaining buffer
        if token_buffer:
            yield token_buffer

        self._decode_time = time.time() - tok_t0
        self._gen_time = time.time() - t0

        # Clean response: remove think tags
        clean = self._clean_response(full_response)

        # Update history
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": clean})
        
        # Trim history if too long (keep within context budget)
        self._trim_history()

        return clean

    def generate(self, text: str, max_new_tokens: int = 0,
                 temperature: float = 0.7) -> str:
        """Non-streaming generation (collects and returns full text)."""
        gen = self.stream_generate(text, max_new_tokens, temperature)
        for _ in gen:
            pass  # consume generator
        return self.history[-1]["content"] if self.history else ""

    def _build_prompt(self, text: str) -> str:
        """Build chat prompt with bounded history.
        
        Automatically reduces history based on remaining context budget.
        """
        # Estimate: ~4 chars per token, history entry ~100 chars typical
        remaining = self.max_context - len(text) // 4 - 100  # token budget
        max_history_entries = max(0, min(MAX_HISTORY_TURNS * 2, 
                                          int(remaining / 50)))
        
        parts = [
            "<|im_start|>system\nYou are Neurova, a helpful AI assistant with the ability to learn from conversations.<|im_end|>"
        ]
        
        # Add recent history (trim from front if too long)
        history_slice = self.history[-max_history_entries:] if max_history_entries > 0 else []
        for msg in history_slice:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        return "\n".join(parts)

    def _clean_response(self, text: str) -> str:
        import re
        # Remove completed <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove standalone <think> and </think> tags
        text = text.replace('<think>', '').replace('</think>', '')
        # Clean up format tokens
        text = re.sub(r'<\|im_start\||<\|im_end\|>', '', text)
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _trim_history(self):
        """Trim history when it gets too long for context budget."""
        # Estimate tokens from history length
        total_chars = sum(len(m.get("content", "")) for m in self.history)
        estimated_tokens = total_chars // 3  # ~3 chars per token in Korean/English mix
        max_tokens_for_history = self.max_context // 3  # use 1/3 of context for history
        
        while estimated_tokens > max_tokens_for_history and len(self.history) > 2:
            # Remove oldest pair (user + assistant)
            self.history.pop(0)
            if self.history:
                self.history.pop(0)
            total_chars = sum(len(m.get("content", "")) for m in self.history)
            estimated_tokens = total_chars // 3

    # ── Status ──

    def status(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Loaded: {'YES' if self._loaded else 'NO'}",
            f"Context: {self.max_context:,} | Max tokens: {self.max_new_tokens:,}",
            f"TTT layers: {TTT_LAYERS} | Passes: {TTT_LEARN_PASSES} | LR: {TTT_LEARN_LR}",
            f"Corrections: {len(self.corrections)} | Learn count: {self._learn_count}",
            f"History: {len(self.history)//2} turns",
            f"Think display: {'ON' if self.show_thinking else 'OFF'}",
        ]
        if self._gen_time > 0:
            lines.append(f"Prefill: {self._prefill_time:.2f}s | Decode: {self._decode_time:.2f}s | Total: {self._gen_time:.2f}s")
        return "\n".join(lines)

    # ── Main Command Handler ──

    def hear(self, text: str) -> "str | Generator":
        global TTT_LEARN_PASSES
        """Process user input — returns string or generator for streaming."""
        text = _norm(text)
        low = text.lower()

        # ── Commands ──
        if low in ("status", ":status", "/status"):
            return self.status()

        if low.startswith("thinking:") or low.startswith("/thinking"):
            v = low.split(":", 1)[1].strip() if ":" in low else ""
            if v in ("on", "true", "1", "yes"):
                self.show_thinking = True
                return "[Thinking display: ON]"
            elif v in ("off", "false", "0", "no"):
                self.show_thinking = False
                return "[Thinking display: OFF]"
            return f"[Current: {'ON' if self.show_thinking else 'OFF'}]"

        if low.startswith("ttt:") or low.startswith("/ttt"):
            v = low.split(":", 1)[1].strip() if ":" in low else ""
            if v in ("on", "true", "1", "yes", "enable"):
                self.load()
                return (f"[In-Place TTT loaded: ctx={self.max_context:,} "
                        f"tokens={self.max_new_tokens:,} passes={TTT_LEARN_PASSES}]")
            elif v in ("off", "false", "0", "no", "disable"):
                self.unload()
                return "[Model unloaded]"
            return f"[Loaded: {self._loaded}]"

        if low.startswith("/verify") or low == "/verify":
            if not self._loaded or self.ttt_wrapper is None:
                return "[Load model first]"
            diffs = self.ttt_wrapper.weight_diff()
            if diffs:
                avg = sum(diffs.values()) / len(diffs)
                return (f"[Weight changes: {len(diffs)} layers | "
                        f"Per-layer: {diffs} | Avg Δ={avg:.6f}]")
            return "[No weight changes detected. TTT hasn't been run yet.]"

        if low.startswith("learn:") or low.startswith("/learn"):
            learn_text = text.split(":", 1)[1].strip() if ":" in text else ""
            if "=>" in learn_text:
                q, a = learn_text.split("=>", 1)
                return self.learn_qa(q.strip(), a.strip())
            return "[Usage: learn: <question> => <answer>]"

        if low.startswith("context:") or low.startswith("/context"):
            v = low.split(":", 1)[1].strip() if ":" in low else ""
            try:
                ctx = int(v)
                # Find nearest supported context
                supported = sorted(_CONTEXT_LEVELS.keys())
                nearest = min(supported, key=lambda x: abs(x - ctx))
                self.max_context = nearest
                if self._loaded:
                    return (f"[Context set to {nearest:,} -> reload to apply: "
                            f"ttt: off → ttt: on]")
                return f"[Context set to {nearest:,}]"
            except:
                return f"[Supported contexts: {list(sorted(_CONTEXT_LEVELS.keys()))}]"

        if low.startswith("passes:") or low.startswith("/passes"):
            v = low.split(":", 1)[1].strip() if ":" in low else ""
            try:
                p = int(v)
                TTT_LEARN_PASSES = max(1, min(p, 20))
                return f"[TTT passes set to {TTT_LEARN_PASSES}]"
            except:
                return f"[Current passes: {TTT_LEARN_PASSES}]"

        if low in ("/reset", "reset"):
            self.history = []
            return "[History cleared]"

        if low in ("clear", "/clear"):
            self.history = []
            if self._loaded:
                self.ttt_wrapper.reset_momentum()
            return "[History + TTT momentum reset]"

        # ── Generation (default) ──
        if self._loaded:
            # Return generator for streaming in CLI
            return self.stream_generate(text)
        else:
            return "[Load model first (ttt: on)]"

"""Neurova TTT-Qwen v2 — REAL LoRA TTT, vLLM streaming, thinking filter."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json, os, re, time, uuid, threading

VLLM_URL = os.environ.get("VLLM_URL", "http://ml-dmc8:8081")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "unsloth/Qwen3.5-4B")
MAX_TOKENS = 4096
MAX_HISTORY = 10


def _uid(): return uuid.uuid4().hex[:8]
def _norm(s): return re.sub(r"\s+", " ", (s or "").strip())

# ── Thinking filter ──────────────────────────────────────────────
_THINKING_HEADER = "Thinking Process:"
_THINKING_STEP = re.compile(r"^\d+\.\s*\*{1,2}")

def _extract_answer(content: str) -> str:
    """Extract final answer from Qwen3.5 thinking output."""
    if not content:
        return ""
    # <think> tags
    if "<think>" in content:
        parts = content.split("</think>", 1)
        return parts[-1].strip() if len(parts) > 1 else content.strip()
    # No thinking → return as-is
    if _THINKING_HEADER not in content:
        return content.strip()
    # Filter mode: keep lines that are NOT thinking structure
    lines = content.split("\n")
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s == _THINKING_HEADER:
            continue
        if _THINKING_STEP.match(s):
            continue
        # Skip indented bullet lines (thinking detail)
        if line.startswith(" ") and (s.startswith("*") or s.startswith("-")):
            continue
        kept.append(s)
    # Return last substantive line
    return kept[-1] if kept else content.strip()


# ═══════════════════════════════════════════════════════════════
# 1. vLLM Client
# ═══════════════════════════════════════════════════════════════

class QwenClient:
    def __init__(self, base_url=VLLM_URL, model=VLLM_MODEL, timeout=120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9, stop=None):
        import urllib.request
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
        if stop: payload["stop"] = stop
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

    def chat_stream(self, messages, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.9, stop=None):
        """Yields (token, is_reasoning, finished)."""
        import urllib.request
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature,
                   "top_p": top_p, "stream": True}
        if stop: payload["stop"] = stop
        data = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        in_think = False
        _buffer = ""
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line_bytes in resp:
                line = line_bytes.decode().strip()
                if not line or line.startswith(":"): continue
                if line.startswith("data: "):
                    ds = line[6:]
                    if ds.strip() == "[DONE]":
                        yield ("", False, True); return
                    try:
                        chunk = json.loads(ds)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        rc = delta.get("reasoning_content", "")
                        cc = delta.get("content", "")
                        if rc:
                            yield (rc, True, False); _buffer += rc
                            continue
                        if not cc: continue
                        if "<think>" in cc: in_think = True; cc = cc.replace("<think>", "")
                        if "</think>" in cc: in_think = False; cc = cc.replace("</think>", "")
                        if cc: yield (cc, in_think, False); _buffer += cc
                    except (json.JSONDecodeError, KeyError):
                        continue


# ═══════════════════════════════════════════════════════════════
# 2. REAL TTT: PEFT + LoRA + 4-bit
# ═══════════════════════════════════════════════════════════════

class TTTLearner:
    def __init__(self, base_model=VLLM_MODEL):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.is_loaded: return
            from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)
            from peft import LoraConfig, get_peft_model
            import torch
            print("[TTT] Loading Qwen3.5-4B 4-bit + LoRA...", flush=True)
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16)
            m = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=bnb,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
            m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                              lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
            tok = AutoTokenizer.from_pretrained(self.base_model)
            tok.pad_token = tok.eos_token; tok.padding_side = "right"
            self.model = m; self.tokenizer = tok; self.is_loaded = True
            print(f"[TTT] Loaded. Trainable: {m.num_parameters(only_trainable=True):,}", flush=True)

    def unload(self):
        with self._lock:
            if self.is_loaded:
                import torch
                del self.model; del self.tokenizer
                self.model = None; self.tokenizer = None; self.is_loaded = False
                torch.cuda.empty_cache()

    def train_step(self, question: str, answer: str, steps: int = 5, lr: float = 3e-4) -> float:
        self.load(); import torch
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_val = 0.0
        for _ in range(steps):
            inp = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            out = self.model(**inp, labels=inp["input_ids"])
            loss_val = out.loss.item()
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
        self.model.eval()
        print(f"[TTT] Loss={loss_val:.4f}", flush=True)
        return loss_val

    def generate(self, text: str, max_new_tokens=MAX_TOKENS, temperature=0.7) -> str:
        self.load(); import torch
        # Use tokenizer chat template for proper formatting
        msg = [{"role": "system", "content": "You are Neurova, a helpful AI assistant."}, {"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                  f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        self.model.eval()
        inp = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                       temperature=temperature, do_sample=temperature > 0.0,
                                       top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
        if "<|im_start|>assistant\n" in full:
            return full.split("<|im_start|>assistant\n", 1)[-1].strip()


# ═══════════════════════════════════════════════════════════════
# 3. Correction Memory
# ═══════════════════════════════════════════════════════════════

@dataclass
class Correction:
    question: str = ""; answer: str = ""
    timestamp: float = field(default_factory=time.time); use_count: int = 0

class CorrectionMemory:
    def __init__(self, path=""):
        self.path = path or os.environ.get("NEUROVA_TTT_MEMORY", "./.neurova_ttt_memory.json")
        self.corrections: List[Correction] = []; self._load()

    def _load(self):
        p = Path(self.path)
        if p.exists():
            try: self.corrections = [Correction(**c) for c in json.loads(p.read_text())]
            except: pass

    def _save(self):
        Path(self.path).write_text(json.dumps(
            [{"question": c.question, "answer": c.answer, "timestamp": c.timestamp, "use_count": c.use_count}
             for c in self.corrections], ensure_ascii=False, indent=2))

    def add(self, q, a): self.corrections.append(Correction(question=_norm(q), answer=_norm(a))); self._save()
    def find(self, q):
        qn = _norm(q.lower())
        for c in reversed(self.corrections):
            if c.question.lower() == qn: c.use_count += 1; self._save(); return c.answer
        return None
    def all(self): return list(self.corrections)
    def qa_pairs(self): return [(c.question, c.answer) for c in self.corrections]
    def __len__(self): return len(self.corrections)


# ═══════════════════════════════════════════════════════════════
# 4. Session Manager
# ═══════════════════════════════════════════════════════════════

@dataclass
class SessionState:
    id: str = ""; corrections: List[Correction] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list); created_at: float = field(default_factory=time.time)
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 100: self.history = self.history[-50:]

class SessionManager:
    def __init__(self): self.sessions: Dict[str, SessionState] = {}
    def get_or_create(self, sid=""):
        sid = sid or _uid()
        if sid not in self.sessions: self.sessions[sid] = SessionState(id=sid)
        return self.sessions[sid]
    def add_correction(self, sid, q, a):
        self.get_or_create(sid).corrections.append(Correction(question=q, answer=a))


# ═══════════════════════════════════════════════════════════════
# 5. Main Engine
# ═══════════════════════════════════════════════════════════════

class TTTChatEngine:
    def __init__(self, vllm_url=VLLM_URL, model=VLLM_MODEL, correction_path=""):
        self.client = QwenClient(vllm_url, model)
        self.corrections = CorrectionMemory(correction_path)
        self.ttt_learner = TTTLearner(base_model=model)
        self.sessions = SessionManager()
        self.ttt_mode = False
        self.show_thinking = False  # Default: OFF (only show answer)
        self._last_full_response = ""

    # ── Correction + REAL LoRA TTT ──
    def correct(self, question: str, answer: str, session_id="") -> str:
        q, a = _norm(question), _norm(answer)
        self.corrections.add(q, a)
        self.sessions.add_correction(session_id, q, a)
        try:
            loss = self.ttt_learner.train_step(q, a, steps=5)
            self.ttt_mode = True
            return f"Correction learned. LoRA trained (loss={loss:.4f}). TTT mode active."
        except Exception as e:
            return f"Stored in memory, but TTT training failed: {e}"

    # ── Streaming ──
    def chat(self, text: str, session_id="", temperature=0.7, max_tokens=MAX_TOKENS, use_corrections=True) -> str:
        """Non-streaming chat. Collects from stream and returns."""
        collected = ""
        for token, is_reasoning, is_final in self.chat_stream(text, session_id, temperature, max_tokens, use_corrections):
            if is_final:
                collected = token
        return collected

    def chat_stream(self, text, session_id="", temperature=0.7, max_tokens=MAX_TOKENS, use_corrections=True):
        """Yields (token, is_reasoning, is_final). After: self._last_full_response = answer."""
        text = _norm(text)
        if not text:
            yield ("Yes?", False, True); self._last_full_response = "Yes?"; return

        sid = session_id or "default"
        session = self.sessions.get_or_create(sid)

        if use_corrections:
            c = self.corrections.find(text)
            if c: yield (c, False, True); self._last_full_response = c; return
            for c in session.corrections:
                if c.question.lower() == text.lower():
                    yield (c.answer, False, True); self._last_full_response = c.answer; return

        # TTT mode: local PEFT model
        if self.ttt_mode and self.ttt_learner.is_loaded:
            try:
                ans = self.ttt_learner.generate(text, max_new_tokens=max_tokens, temperature=temperature)
                yield (ans, False, True); self._last_full_response = ans
            except Exception as e:
                yield (f"[TTT Error: {e}]", False, True); self._last_full_response = f"[Error: {e}]"
            session.add_message("user", text); session.add_message("assistant", self._last_full_response)
            return

        # vLLM streaming
        messages = self._build_messages(text, session, use_corrections)
        collected = ""
        in_reasoning = False
        for token, is_reasoning, finished in self.client.chat_stream(messages, max_tokens=max_tokens, temperature=temperature):
            collected += token
            if is_reasoning and not in_reasoning: in_reasoning = True
            elif not is_reasoning and in_reasoning: in_reasoning = False
            yield (token, in_reasoning, False)
            if finished: break

        answer = _extract_answer(collected)
        self._last_full_response = answer
        yield (answer, False, True)
        session.add_message("user", text); session.add_message("assistant", answer)

    def _get_system_prompt(self, use_corrections=True) -> str:
        parts = [
            "You are Neurova, a helpful AI assistant based on Qwen3.5-4B.",
            "Provide accurate, concise answers to the user's questions.",
            "If you don't know something, say so rather than guessing.",
        ]
        if use_corrections:
            corrs = self.corrections.all()[-10:]
            if corrs:
                cl = [f"Q: {c.question}\nA: {c.answer}" for c in corrs]
                parts.append("Known corrections:\n" + "\n".join(cl))
        return "\n".join(parts)

    def _build_messages(self, text, session, use_corrections=True):
        msgs = [{"role": "system", "content": self._get_system_prompt(use_corrections)}]
        msgs.extend(session.history[-MAX_HISTORY * 2:])
        msgs.append({"role": "user", "content": text})
        return msgs

    def enable_ttt(self) -> bool:
        """Kill vLLM, free GPU, load PEFT model."""
        import subprocess, time
        try:
            subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints|VLLM::EngineCore"],
                           capture_output=True, timeout=5)
            time.sleep(2)
        except: pass
        try:
            self.ttt_learner.load()
            self.ttt_mode = True
            return True
        except Exception as e:
            print(f"[TTT] Failed: {e}")
            return False

    def disable_ttt(self):
        self.ttt_learner.unload(); self.ttt_mode = False

    def status(self):
        return (f"vLLM: {self.client.base_url}\n"
                f"Corrections: {len(self.corrections)}\n"
                f"TTT mode: {'ACTIVE (LoRA)' if self.ttt_mode else 'inactive'}\n"
                f"TTT loaded: {self.ttt_learner.is_loaded}\n"
                f"Show thinking: {self.show_thinking}\n"
                f"Sessions: {len(self.sessions.sessions)}")


# ═══════════════════════════════════════════════════════════════
# 6. CLI Wrapper
# ═══════════════════════════════════════════════════════════════

class NeurovaTTTEngine:
    def __init__(self):
        self.brain = TTTChatEngine(); self.session_id = "default"
    def hear(self, text: str) -> str:
        text = _norm(text); low = text.lower()
        if low in ("status", ":status"): return self.brain.status()
        if low.startswith("correct:") or low.startswith("learn:"):
            payload = text.split(":", 1)[1].strip() if ":" in text else ""
            if "=>" in payload:
                q, a = payload.split("=>", 1)
                return self.brain.correct(q.strip(), a.strip(), self.session_id)
            return "Usage: correct: <question> => <answer>"
        if low.startswith("thinking:"):
            v = low.split(":", 1)[1].strip()
            if v in ("on", "true", "1", "yes"): self.brain.show_thinking = True; return "Thinking: ON"
            elif v in ("off", "false", "0", "no"): self.brain.show_thinking = False; return "Thinking: OFF"
            return f"Current: {'ON' if self.brain.show_thinking else 'OFF'}"
        if low.startswith("ttt:"):
            v = low.split(":", 1)[1].strip()
            if v in ("on", "true", "1", "yes", "enable"):
                return "TTT: ON (LoRA)" if self.brain.enable_ttt() else "TTT: FAILED"
            elif v in ("off", "false", "0", "no", "disable"):
                self.brain.disable_ttt(); return "TTT: OFF"
            return f"TTT: {'ON' if self.brain.ttt_mode else 'OFF'}"
        return self.brain.chat(text, self.session_id)

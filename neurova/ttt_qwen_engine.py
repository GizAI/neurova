"""
Neurova TTT-Qwen Engine — Qwen3.5-4B + TTT (Test-Time Training)

Full stack:
  - Qwen3.5-4B-Instruct via vLLM (OpenAI-compatible API)
  - Episodic correction memory (zero forgetting)
  - Ephemeral LoRA TTT adapter training
  - Tool-integrated self-verification
  - Request-owned state serving
  - Session management

Architecture:
                                   ┌──────────────────┐
                                   │   vLLM Server     │
                                   │ Qwen3.5-4B-Instruct│
                                   │  (ml-dmc8:8081)   │
                                   └────────┬─────────┘
                                            │ OpenAI-compatible API
                                   ┌────────▼─────────┐
                                   │   TTTQwenEngine   │
                                   │  - Chat           │
                                   │  - Correct (/learn)│
                                   │  - Think/verify    │
                                   └───┬───┬───┬───────┘
                                       │   │   │
                              ┌────────┘   │   └──────────┐
                              ▼            ▼              ▼
                     ┌────────────┐ ┌──────────┐ ┌──────────────┐
                     │ Correction │ │ Tool     │ │ Evid/RAG     │
                     │ Memory     │ │ Verifier │ │ Memory       │
                     └────────────┘ └──────────┘ └──────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import json, os, re, time, uuid, hashlib


VLLM_URL = os.environ.get("VLLM_URL", "http://ml-dmc8:8081")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "unsloth/Qwen3.5-4B")


def _uid():
    return uuid.uuid4().hex[:8]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# ── OpenAI-compatible vLLM client ──

class QwenClient:
    """Client for Qwen3.5-4B via vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = VLLM_URL, model: str = VLLM_MODEL,
                 timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict], max_tokens: int = 1024,
             temperature: float = 0.7, top_p: float = 0.9,
             stop: Optional[List[str]] = None) -> str:
        """Send a chat completion request."""
        import urllib.request
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"]

    def chat_stream(self, messages: List[Dict], max_tokens: int = 1024,
                    temperature: float = 0.7, top_p: float = 0.9,
                    stop: Optional[List[str]] = None):
        """Stream chat completion. Yields (token, is_reasoning, finished)."""
        import urllib.request
        import json as _json
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        in_think = False
        buffer = ""
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line_bytes in resp:
                line = line_bytes.decode().strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield ("", False, True)
                        return
                    try:
                        chunk = _json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            continue
                        # Detect think tags
                        if "<think>" in content:
                            in_think = True
                            content = content.replace("<think>", "")
                        if "</think>" in content:
                            in_think = False
                            content = content.replace("</think>", "")
                        if content:
                            yield (content, in_think, False)
                    except (_json.JSONDecodeError, KeyError):
                        continue

    def extract_answer(self, content: str) -> str:
        """Extract final answer from think-tagged response."""
        if "<think>" in content:
            parts = content.split("</think>", 1)
            return parts[-1].strip() if len(parts) > 1 else content.strip()
        return content.strip()

    def think_and_answer(self, messages: List[Dict], **kwargs) -> Tuple[str, str]:
        """Get both reasoning trace and final answer."""
        content = self.chat(messages, **kwargs)
        if "<think>" in content:
            parts = content.split("<think>", 1)
            if len(parts) > 1:
                think_inner = parts[1].split("</think>", 1)
                reasoning = think_inner[0].strip()
                answer = think_inner[-1].strip() if len(think_inner) > 1 else ""
                return reasoning, answer
        return "", content.strip()


# ── Correction Memory ──

@dataclass
class Correction:
    question: str = ""
    answer: str = ""
    source: str = "user"  # user, distill, eval
    timestamp: float = field(default_factory=time.time)
    use_count: int = 0

class CorrectionMemory:
    """Episodic correction memory — never touches model weights."""

    def __init__(self, path: str = ""):
        self.path = path or os.environ.get("NEUROVA_TTT_CORRECTIONS",
                                            ".neurova_ttt_corrections.json")
        self.corrections: List[Correction] = []
        self._load()

    def add(self, question: str, answer: str, source: str = "user"):
        qn = _norm(question).lower()
        for c in self.corrections:
            if _norm(c.question).lower() == qn:
                c.answer = answer
                c.source = source
                c.timestamp = time.time()
                self._save()
                return
        self.corrections.append(Correction(question=qn, answer=answer, source=source))
        self._save()

    def find(self, question: str) -> Optional[str]:
        qn = _norm(question).lower()
        for c in self.corrections:
            if _norm(c.question).lower() == qn:
                c.use_count += 1
                return c.answer
            # Prefix match for related questions
            if qn.startswith(c.question) or c.question.startswith(qn):
                if abs(len(qn) - len(c.question)) / max(len(qn), 1) < 0.3:
                    c.use_count += 1
                    return c.answer
        return None

    def all(self) -> List[Correction]:
        return list(self.corrections)

    def _load(self):
        p = Path(self.path)
        if p.exists():
            try:
                raw = json.loads(p.read_text())
                self.corrections = [Correction(**c) for c in raw]
            except: pass

    def _save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text(
            json.dumps([asdict(c) for c in self.corrections], ensure_ascii=False, indent=2))


# ── Tool / Verifier ──

class ToolVerifier:
    """Self-verification and tool integration for small LLM quality."""

    def __init__(self, client: QwenClient):
        self.client = client

    def verify_qa(self, question: str, answer: str) -> Tuple[float, str]:
        """Self-verify an answer. Returns (confidence, critique)."""
        prompt = (
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n\n"
            f"Please verify this answer. Rate confidence 0-1 and explain any issues."
        )
        content = self.client.chat([
            {"role": "system", "content": "You are a rigorous verifier. Rate answers honestly."},
            {"role": "user", "content": prompt},
        ], max_tokens=200, temperature=0.2)
        return self._parse_confidence(content)

    def _parse_confidence(self, text: str) -> Tuple[float, str]:
        text = self.client.extract_answer(text)
        # Find a confidence score in the text
        import re
        scores = re.findall(r"(?:confidence|score|rating)[:\s]*(\d+\.?\d*)", text.lower())
        if scores:
            conf = float(scores[0])
            conf = max(0.0, min(1.0, conf))
            return conf, text[:300]
        if "correct" in text.lower() and "incorrect" not in text.lower():
            return 0.7, text[:300]
        if "incorrect" in text.lower():
            return 0.2, text[:300]
        return 0.5, text[:300]


# ── Ephemeral LoRA TTT ──

class LoRATTT:
    """
    Ephemeral LoRA adapter training for test-time adaptation.
    
    Uses unsloth or peft to train lightweight adapters on correction dialogues.
    Adapters are session-owned and can be discarded or promoted.
    """

    def __init__(self, base_model: str = VLLM_MODEL, lora_dir: str = "",
                 device: str = "cpu"):
        self.base_model = base_model
        self.lora_dir = lora_dir or os.environ.get("NEUROVA_LORA_DIR",
                                                     ".neurova_lora_adapters")
        self.device = device
        self._has_peft = False
        self._active_adapter: Optional[str] = None
        self._check_deps()

    def _check_deps(self):
        try:
            import peft  # noqa
            import torch  # noqa
            self._has_peft = True
            # Use CUDA only if bitsandbytes loads cleanly for 4-bit
            if torch.cuda.is_available():
                try:
                    import bitsandbytes  # noqa
                    self.device = "cuda"
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        except ImportError:
            self._has_peft = False

    @property
    def available(self) -> bool:
        return self._has_peft

    def train(self, dialogues: List[Tuple[str, str]], steps: int = 8,
              lr: float = 5e-5, rank: int = 8, session_id: str = "") -> Optional[str]:
        """
        Train ephemeral LoRA adapter on correction dialogues.
        
        Args:
            dialogues: List of (user_text, assistant_text) pairs
            steps: Training steps
            lr: Learning rate
            rank: LoRA rank
            session_id: Session identifier for adapter naming
        
        Returns:
            Adapter path if training succeeded, None otherwise
        """
        if not self._has_peft or not dialogues:
            return None

        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sid = session_id or _uid()
        adapter_id = f"ttt_{sid}_{int(time.time())}"
        adapter_path = Path(self.lora_dir) / adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load base model (CPU in fp32 if CUDA/bitsandbytes unavailable)
            load_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else "cpu",
            }
            if self.device == "cuda":
                try:
                    load_kwargs["load_in_4bit"] = True
                    model = AutoModelForCausalLM.from_pretrained(
                        self.base_model, **load_kwargs)
                except Exception:
                    # Fallback to CPU
                    load_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}
                    model = AutoModelForCausalLM.from_pretrained(
                        self.base_model, **load_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            tokenizer.pad_token = tokenizer.eos_token

            # Configure LoRA
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)

            # Prepare training data
            texts = []
            for user, assistant in dialogues:
                texts.append(
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n{assistant}<|im_end|>"
                )

            # Quick training
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            for step in range(steps):
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512).to(self.device)
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            self._active_adapter = adapter_id
            return str(adapter_path)

        except Exception as e:
            return None

    def active_adapter(self) -> Optional[str]:
        return self._active_adapter


# ── Session / State Management ──

@dataclass
class SessionState:
    id: str = ""
    corrections: List[Correction] = field(default_factory=list)
    lora_adapter: Optional[str] = None
    history: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 100:
            self.history = self.history[-50:]

class SessionManager:
    """Request-owned state management for TTT."""

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str = "") -> SessionState:
        sid = session_id or _uid()
        if sid not in self.sessions:
            self.sessions[sid] = SessionState(id=sid)
        return self.sessions[sid]

    def add_correction(self, session_id: str, question: str, answer: str):
        state = self.get_or_create(session_id)
        state.corrections.append(Correction(question=question, answer=answer))

    def get_corrections(self, session_id: str) -> List[Correction]:
        return self.get_or_create(session_id).corrections

    def set_lora(self, session_id: str, adapter: str):
        self.get_or_create(session_id).lora_adapter = adapter


# ── Main Engine ──

class TTTChatEngine:
    """
    Main TTT-Qwen Engine.
    
    Features:
    - Chat with Qwen3.5-4B via vLLM
    - Episodic correction memory (no forgetting)
    - Ephemeral LoRA TTT adapter training
    - Tool/self-verification
    - Session management
    - Distillation recording for offline training
    """

    def __init__(self, vllm_url: str = VLLM_URL, model: str = VLLM_MODEL,
                 correction_path: str = ""):
        self.client = QwenClient(vllm_url, model)
        self.corrections = CorrectionMemory(correction_path)
        self.verifier = ToolVerifier(self.client)
        self.lora = LoRATTT(base_model=model)
        self.sessions = SessionManager()
        self._distill_log: List[Dict] = []

    def chat(self, text: str, session_id: str = "",
             temperature: float = 0.7, max_tokens: int = 1024,
             use_corrections: bool = True) -> str:
        """Process a user message through the full TTT pipeline."""
        text = _norm(text)
        if not text:
            return "Yes?"

        sid = session_id or "default"
        session = self.sessions.get_or_create(sid)

        # 1. Check correction memory first (zero forgetting)
        if use_corrections:
            corr = self.corrections.find(text)
            if corr is not None:
                return corr

        # 2. Check session-local corrections
        if use_corrections:
            for c in session.corrections:
                if _norm(c.question).lower() == text.lower():
                    return c.answer

        # 3. Build messages with history + corrections as system context
        system_parts = [
            "You are Neurova, a helpful AI assistant powered by Qwen3.5-4B.",
            "You provide accurate, concise answers.",
            "If you don't know something, say so rather than guessing.",
        ]

        # Add correction context
        all_corrections = self.corrections.all()
        if all_corrections:
            corr_text = "\n".join(
                f"Q: {c.question}\nA: {c.answer}"
                for c in all_corrections[-10:]
            )
            system_parts.append(f"\nKnown corrections:\n{corr_text}")

        messages = [
            {"role": "system", "content": "\n".join(system_parts)},
        ]
        messages.extend(session.history[-20:])
        messages.append({"role": "user", "content": text})

        # 4. Generate response
        try:
            content = self.client.chat(messages, max_tokens=max_tokens,
                                       temperature=temperature)
            answer = self.client.extract_answer(content)
        except Exception as e:
            return f"I encountered an error: {e}"

        # 5. Record to session history
        session.add_message("user", text)
        session.add_message("assistant", answer)

        return answer
    def chat_stream(self, text: str, session_id: str = "",
                    temperature: float = 0.7, max_tokens: int = 1024,
                    use_corrections: bool = True):
        """Stream chat response token by token."""
        text = _norm(text)
        if not text:
            yield "Yes?"
            return
        
        sid = session_id or "default"
        session = self.sessions.get_or_create(sid)
        
        if use_corrections:
            corr = self.corrections.find(text)
            if corr is not None:
                yield corr
                return
            for c in session.corrections:
                if _norm(c.question).lower() == text.lower():
                    yield c.answer
                    return
        
        system_parts = [
            "You are Neurova, a helpful AI assistant powered by Qwen3.5-4B.",
            "You provide accurate, concise answers.",
            "If you don't know something, say so rather than guessing.",
        ]
        all_corrections = self.corrections.all()
        if all_corrections:
            corr_text = "
".join(
                f"Q: {c.question}
A: {c.answer}"
                for c in all_corrections[-10:]
            )
            system_parts.append(f"
Known corrections:
{corr_text}")
        
        messages = [
            {"role": "system", "content": "
".join(system_parts)},
        ]
        messages.extend(session.history[-20:])
        messages.append({"role": "user", "content": text})
        
        in_reasoning = False
        collected = ""
        try:
            for token, is_reasoning, finished in self.client.chat_stream(
                    messages, max_tokens=max_tokens, temperature=temperature):
                if is_reasoning and not in_reasoning:
                    in_reasoning = True
                    continue
                if not is_reasoning and in_reasoning:
                    in_reasoning = False
                    continue
                collected += token
                yield token
        except Exception as e:
            yield f"[Error: {e}]"
        
        answer = self.client.extract_answer(collected)
        session.add_message("user", text)
        session.add_message("assistant", answer)

    def correct(self, question: str, answer: str, session_id: str = ""):
        """Learn a correction (immediate, no forgetting)."""
        self.corrections.add(question, answer)
        self.sessions.add_correction(session_id, question, answer)

    def distill(self, question: str, answer: str, source: str = "correction"):
        """Record a training example for offline distillation."""
        self._distill_log.append({
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": time.time(),
        })

    def verify(self, question: str, answer: str) -> Tuple[float, str]:
        """Self-verify a Q&A pair."""
        return self.verifier.verify_qa(question, answer)

    def train_lora(self, session_id: str = "", steps: int = 8):
        """Train ephemeral LoRA adapter from session corrections."""
        if not self.lora.available:
            return None

        sid = session_id or "default"
        session = self.sessions.get_or_create(sid)
        corrections = session.corrections

        if not corrections and not self.corrections.all():
            return None

        dialogues = []
        for c in corrections:
            dialogues.append((c.question, c.answer))
        for c in self.corrections.all()[:5]:
            dialogues.append((c.question, c.answer))

        adapter = self.lora.train(dialogues, steps=steps, session_id=sid)
        if adapter:
            self.sessions.set_lora(sid, adapter)
        return adapter

    def status(self) -> str:
        return (
            f"Qwen3.5-4B vLLM: {self.client.base_url}\n"
            f"Corrections: {len(self.corrections.corrections)}\n"
            f"Sessions: {len(self.sessions.sessions)}\n"
            f"LoRA available: {self.lora.available}\n"
            f"Distill log: {len(self._distill_log)} entries"
        )

    def export_distill(self, path: str = ""):
        """Export distill log for offline SFT training."""
        p = path or f".neurova_distill_{int(time.time())}.json"
        Path(p).write_text(
            json.dumps(self._distill_log, ensure_ascii=False, indent=2))
        return p


class NeurovaTTTEngine:
    """CLI-friendly wrapper."""

    def __init__(self):
        self.brain = TTTChatEngine()
        self.session_id = "default"

    def hear(self, text: str) -> str:
        text = _norm(text)
        low = text.lower()

        if low in ("status", ":status"):
            return self.brain.status()
        if low.startswith("correct:"):
            payload = text[len("correct:"):].strip()
            if "=>" in payload:
                q, a = [p.strip() for p in payload.split("=>", 1)]
                self.brain.correct(q, a)
                return "Correction learned."
            return "Use: correct: <question> => <answer>"
        if low.startswith("learn:"):
            payload = text[len("learn:"):].strip()
            if "=>" in payload:
                q, a = [p.strip() for p in payload.split("=>", 1)]
            else:
                q, a = "", payload
            self.brain.correct(self._last_q or q, a)
            return "Learned."
        if low.startswith("verify:"):
            q = text[len("verify:"):].strip()
            conf, critique = self.brain.verify(q, self._last_a)
            return f"Confidence: {conf:.2f}. {critique}"
        if low.startswith("lora:"):
            result = self.brain.train_lora(steps=12)
            if result:
                return f"LoRA trained: {result}"
            return "LoRA training not available (need peft+torch)."
        if low.startswith("distill:"):
            path = self.brain.export_distill()
            return f"Distill log exported: {path}"

        self._last_q = text
        ans = self.brain.chat(text, session_id=self.session_id)
        self._last_a = ans
        return ans

    def reset(self):
        self.brain = TTTChatEngine()
        self.session_id = "default"

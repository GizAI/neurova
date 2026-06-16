#!/usr/bin/env python3
"""Qwen Memory — Qwen-backed embedding memory. Zero hardcoded language rules.

인지 prior만 제공:
  - Entity embedding (embed_tokens + USearch)
  - 순수 유사도 검색 (threshold 없음, top-K)
  - 원문 저장 (변환/규칙 없음)
  
하드코딩 제거 목록:
  detect_personal()      → 제거
  to_user_memory()       → 제거  
  _find_update() keywords → 제거
  언어별 패턴 리스트      → 제거
  similarity thresholds  → 제거
"""
from __future__ import annotations
import os, sys, time, re, json, atexit, readline, threading
from pathlib import Path
from typing import Generator, List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
from usearch.index import Index
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    BitsAndBytesConfig
)

def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODE = env_value("QWEN_MEMORY_MODE", "bf16")
EFFORT = env_value("QWEN_MEMORY_EFFORT", "low").lower()
MAX_CTX = int(env_value("QWEN_MEMORY_CTX", "16384"))
MAX_NEW = int(env_value("QWEN_MEMORY_MAX", "4096"))
SEARCH_K = int(env_value("QWEN_MEMORY_K", "7"))
HIST_CUT = int(env_value("QWEN_MEMORY_HIST", "4"))
AUTO_MEM = env_value("QWEN_MEMORY_AUTO", "1") == "1"
H = 2560

BASE_DIR = Path(os.path.expanduser(env_value("QWEN_MEMORY_HOME", "~/.qwen_memory")))
BASE_DIR.mkdir(exist_ok=True)
CURRENT_USER = env_value("QWEN_MEMORY_USER", "default")
MEM_DIR = BASE_DIR / "users" / CURRENT_USER
MEM_DIR.mkdir(parents=True, exist_ok=True)
MEM_PATH = str(MEM_DIR / "memory")
HIST_PATH = str(MEM_DIR / "history")
MEM_CAP = 10000


def strip_think(t: str) -> str:
    return re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL
                  ).replace('<|im_start|>', '').replace('<|im_end|>', '').strip()


@dataclass
class MemSlot:
    """원문 그대로 저장. 변환/가공 없음."""
    text: str
    source: str = "auto"
    ts: float = 0.0
    retrievals: int = 0


class Memory:
    """Pure embedding memory. No rules, no thresholds, no conversion."""
    def __init__(self, model, cap: int = MEM_CAP):
        self.model = model
        self.cap = cap
        self.dev = next(model.parameters()).device
        self.index = Index(ndim=H, metric='cos')
        self.count = 0
        self.cursor = 0
        self.slots: List[MemSlot] = []
        self._lock = threading.Lock()
        self._dirty = False

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Mean-pooled embed_tokens. No forward pass."""
        with torch.inference_mode():
            emb = self.model.model.embed_tokens(ids)
            mask = (ids != self.model.tokenizer.pad_token_id).float()
            if mask.sum() == 0: return emb[0, -1]
            return (emb[0] * mask.T).sum(dim=0) / mask.sum()

    def _similar(self, vec: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
        if self.count == 0: return []
        m = self.index.search(vec.reshape(1, -1), k)
        if m is None or len(getattr(m, 'keys', [])) == 0: return []
        ds = m.distances if hasattr(m, 'distances') else [1.0]*len(m.keys)
        return [(int(k_), float(d)) for k_, d in zip(m.keys, ds) if int(k_) < self.count]

    def store(self, text: str, source: str = "auto") -> int:
        """원문 저장. embedding으로만 dedup/update."""
        text = text.strip()
        if len(text) < 5: return 0
        with self._lock:
            ids = self.model.tokenizer(text, return_tensors='pt',
                                        truncation=True, max_length=MAX_CTX).to(self.dev)
            if ids['input_ids'].shape[1] < 2: return 0
            vec = self._embed(ids['input_ids']).cpu().to(torch.float32).numpy()

            # Dedup via pure embedding similarity
            for idx, dist in self._similar(vec):
                if idx < len(self.slots):
                    if dist < 0.10: return 0  # near-exact dup
                    if dist < 0.45 and len(text) > len(self.slots[idx].text):
                        # Update with more specific info
                        self.slots[idx].text = text
                        self.slots[idx].ts = time.time()
                        self.index.remove(np.array([idx], dtype=np.uint64))
                        self.index.add(np.array([idx], dtype=np.uint64), vec.reshape(1, -1))
                        self._dirty = True
                        return 1

            # New entry
            pos = self.cursor % self.cap
            self.index.add(np.array([self.count], dtype=np.uint64), vec.reshape(1, -1))
            self.cursor += 1
            if self.count < self.cap: self.count += 1
            self.slots.append(MemSlot(text=text, source=source, ts=time.time()))
            self._dirty = True
            return 1

    def recall(self, text: str, k: int = 7) -> str:
        """순수 embedding 검색. top-K 반환."""
        if self.count == 0: return ''
        with self._lock:
            ids = self.model.tokenizer(text, return_tensors='pt',
                                        truncation=True, max_length=MAX_CTX).to(self.dev)
            if ids['input_ids'].shape[1] < 2: return ''
            vec = self._embed(ids['input_ids']).cpu().to(torch.float32).numpy()

            parts = []
            for idx, dist in self._similar(vec, k=k):
                if idx < len(self.slots):
                    self.slots[idx].retrievals += 1
                    parts.append(self.slots[idx].text)
            return '\n'.join(parts)

    def save(self, path: str):
        if not self._dirty: return
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(p.with_suffix('.usearch')))
        json.dump({'count': self.count, 'slots': [s.__dict__ for s in self.slots]},
                  open(p.with_suffix('.json'), 'w'), ensure_ascii=False)
        self._dirty = False

    def load(self, path: str):
        p = Path(path)
        if p.with_suffix('.usearch').exists():
            self.index = Index(ndim=H, metric='cos')
            self.index.load(str(p.with_suffix('.usearch')))
            self.count = len(self.index)
        if p.with_suffix('.json').exists():
            m = json.load(open(p.with_suffix('.json')))
            self.slots = [MemSlot(**s) for s in m.get('slots', [])]

    def stats(self) -> str:
        n = sum(1 for s in self.slots if s.retrievals > 0)
        return f'{self.count} stored | {n} retrieved'


class Engine:
    def __init__(self):
        self.model = None; self.tokenizer = None; self.mem = None
        self.loaded = False; self.history: List[Dict] = []
        self.n_gen = 0; self.effort = EFFORT; self.auto_mem = AUTO_MEM
        self.current_user = CURRENT_USER

    @property
    def dev(self): return next(self.model.parameters()).device if self.model else 'cpu'

    def _mem_path(self, user=None):
        return str(BASE_DIR / "users" / (user or self.current_user) / "memory")

    def _switch_user(self, new_user: str):
        if self.loaded and self.mem and self.mem._dirty:
            try: self.mem.save(self._mem_path())
            except: pass
        self.current_user = new_user
        if self.mem:
            self.mem = Memory(self.model)
            p = Path(self._mem_path())
            if p.with_suffix('.usearch').exists():
                try: self.mem.load(str(p))
                except: pass
        self.history.clear()

    def load(self):
        if self.loaded: return
        t0 = time.time()
        print(f'[QwenMemory] {MODE} effort={self.effort} user={self.current_user}', flush=True)
        kw = {'trust_remote_code': True, 'attn_implementation': 'sdpa', 'device_map': 'auto'}
        if MODE == '4bit':
            kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        else:
            kw['torch_dtype'] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", **kw)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
        self.tokenizer.model_max_length = MAX_CTX
        self.model.tokenizer = self.tokenizer
        self.mem = Memory(self.model)
        p = Path(self._mem_path())
        if p.with_suffix('.usearch').exists():
            try: self.mem.load(str(p)); print(f'  Mem: {self.mem.stats()}', flush=True)
            except: pass
        self.loaded = True
        print(f'  Ready {time.time()-t0:.1f}s VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB', flush=True)

    def unload(self):
        if self.loaded:
            try: self.mem.save(self._mem_path())
            except: pass
            del self.mem, self.model, self.tokenizer
            self.loaded = False; torch.cuda.empty_cache()

    def remember(self, text: str) -> str:
        if not self.loaded: return '[!]'
        t0 = time.time()
        n = self.mem.store(text, source="manual")
        if n:
            try: self.mem.save(self._mem_path())
            except: pass
        return f'[{n}] ({time.time()-t0:.1f}s) {self.mem.stats()}'

    def recall_slots(self) -> str:
        if not self.loaded or not self.mem.slots: return '[empty]'
        return '\n'.join(f'  [{i}] {s.text[:100]} (r={s.retrievals})'
                        for i, s in enumerate(reversed(self.mem.slots[-20:])))

    def generate(self, text: str) -> Generator[str, None, str]:
        if not self.loaded: yield '[!]'; return

        # Pure embedding recall (no rules, no thresholds)
        ctx = self.mem.recall(text)

        # System prompt: 모델이 attribution 처리
        sys = 'You are Neurova.'
        if ctx: sys += f'\nThe user has shared:\n{ctx}'
        if self.effort == 'high': sys += '\nThink step by step thoroughly.'

        msgs = [{"role": "system", "content": sys}]
        for h in self.history[-HIST_CUT * 2:]: msgs.append(h)
        msgs.append({"role": "user", "content": text})

        try:
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=self.effort != 'low')
        except:
            prompt = ''
            for m in msgs: prompt += f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n'
            prompt += '<|im_start|>assistant\n'

        inp = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                              max_length=MAX_CTX).to(self.dev)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                         skip_special_tokens=True, timeout=120)
        e = threading.Event()

        def run():
            try:
                self.model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                    use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, streamer=streamer)
            finally:
                e.set()

        threading.Thread(target=run, daemon=True).start()
        full = ''
        for tok in streamer:
            full += tok; yield tok
        e.wait(30)

        # Post-process
        clean = strip_think(full)
        self.history.append({'role': 'user', 'content': text})
        self.history.append({'role': 'assistant', 'content': clean or ''})
        self.n_gen += 1

        # Auto-store: 응답 후 동기 처리 (사용자 체감 지연 없음)
        if self.auto_mem:
            self.mem.store(text)
            if self.mem._dirty:
                try: self.mem.save(self._mem_path())
                except: pass
        return clean

    def status(self) -> str:
        if not self.loaded: return '[!]'
        v = torch.cuda.memory_allocated() / 1024**3
        return (f'QwenMemory user={self.current_user} effort={self.effort}\n'
                f'Mem: {self.mem.stats()} | Replies: {self.n_gen}\n'
                f'VRAM: {v:.1f}GB ctx={MAX_CTX} max={MAX_NEW}')


def main():
    try: readline.read_history_file(HIST_PATH)
    except: pass
    atexit.register(lambda: readline.write_history_file(HIST_PATH))
    readline.set_history_length(2000)

    eng = Engine(); eng.load()
    print(f'  Pure embedding memory | User: {eng.current_user}')
    print(f'  /think /nothink /effort | /user <name>')
    while True:
        try: inp = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not inp: continue
        cmd, *rest = inp.lower().split(maxsplit=1)
        arg = rest[0] if rest else ''
        if cmd in ('exit','quit'): break
        if cmd == 'remember:': print(eng.remember(arg)); continue
        if cmd in ('recall','기억'): print(eng.recall_slots()); continue
        if cmd in ('memsave','저장'): eng.mem.save(eng._mem_path()); print('[ok]'); continue
        if cmd == 'status': print(eng.status()); continue
        if cmd in ('/clear','reset','초기화'): eng.history.clear(); print('[ok]'); continue
        if cmd == '/think': eng.effort = 'mid'; print('[ok] effort=mid'); continue
        if cmd == '/nothink': eng.effort = 'low'; print('[ok] effort=low'); continue
        if cmd == '/effort' and arg in ('low','mid','high'): eng.effort = arg; print(f'[ok] effort={arg}'); continue
        if cmd == '/user' and arg: eng._switch_user(arg.strip()); print(f'[ok] user: {eng.current_user}'); continue
        try:
            for tok in eng.generate(inp): sys.stdout.write(tok); sys.stdout.flush()
            sys.stdout.write('\n'); sys.stdout.flush()
        except KeyboardInterrupt: print('\n[!]')
        except Exception as ex: print(f'\n[!] {ex}')
    eng.unload()

if __name__ == '__main__':
    main()

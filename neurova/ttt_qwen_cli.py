#!/usr/bin/env python3
"""Neurova TTT-Qwen CLI — Real TTT, streaming, thinking toggle."""
import os, sys, readline, atexit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurova.ttt_qwen_engine import NeurovaTTTEngine, VLLM_URL, VLLM_MODEL, MAX_TOKENS

def main():
    hist = os.path.expanduser("~/.neurova_ttt_qwen_history")
    try: readline.read_history_file(hist)
    except FileNotFoundError: pass
    readline.set_history_length(2000)
    atexit.register(readline.write_history_file, hist)

    print("=" * 68)
    print(f"  Neurova TTT-Qwen v2 — {VLLM_MODEL} + Real TTT (LoRA)")
    print(f"  vLLM: {VLLM_URL}  |  Max tokens: {MAX_TOKENS}")
    print(f"  Commands:")
    print(f"    learn: <Q> => <A>     REAL TTT — trains LoRA weights")
    print(f"    ttt: on/off           Toggle TTT mode (PEFT LoRA)")
    print(f"    thinking: on/off      Toggle reasoning display")
    print(f"    status                Show engine status")
    print("=" * 68)

    engine = NeurovaTTTEngine()
    brain = engine.brain
    print(f"\n[System] Ready.\n")

    while True:
        try:
            inp = input(">>> ").strip()
        except EOFError:
            print(); break
        if not inp: continue
        if inp.lower() in ("exit", "quit", ":q"): break

        try:
            low = inp.lower()
            if low in ("status", ":status"):
                print(f"[TTT-Qwen]\n{brain.status()}\n")
            elif low.startswith("correct:") or low.startswith("learn:"):
                payload = inp.split(":", 1)[1].strip() if ":" in inp else ""
                if "=>" in payload:
                    q, a = payload.split("=>", 1)
                    print(f"[TTT-Qwen] {brain.correct(q.strip(), a.strip(), engine.session_id)}\n")
                else:
                    print("[TTT-Qwen] Usage: learn: <question> => <answer>\n")
            elif low.startswith("ttt:"):
                val = low.split(":", 1)[1].strip()
                if val in ("on", "true", "1", "yes", "enable"):
                    print("[TTT] Loading model... ", end="", flush=True)
                    ok = brain.enable_ttt()
                    print(f"{'OK' if ok else 'FAILED'}", flush=True)
                    print(f"[TTT-Qwen] TTT: {'ON (LoRA)' if ok else 'OFF'}\n")
                elif val in ("off", "false", "0", "no", "disable"):
                    brain.disable_ttt(); print("[TTT-Qwen] TTT: OFF (using vLLM)\n")
                else:
                    print(f"[TTT-Qwen] TTT: {'ON' if brain.ttt_mode else 'OFF'}\n")
            elif low.startswith("thinking:"):
                v = low.split(":", 1)[1].strip()
                if v in ("on", "true", "1", "yes"):
                    brain.show_thinking = True; print("[TTT-Qwen] Thinking: ON\n")
                elif v in ("off", "false", "0", "no"):
                    brain.show_thinking = False; print("[TTT-Qwen] Thinking: OFF\n")
                else:
                    print(f"[TTT-Qwen] Current: {'ON' if brain.show_thinking else 'OFF'}\n")
            else:
                # ── Chat ──
                if brain.ttt_mode and brain.ttt_learner.is_loaded:
                    # TTT mode: PEFT LoRA model
                    print("[TTT-Qwen] ", end="", flush=True)
                    ans = brain.ttt_learner.generate(inp)
                    print(ans)
                    s = brain.sessions.get_or_create(engine.session_id)
                    s.add_message("user", inp); s.add_message("assistant", ans)
                elif brain.show_thinking:
                    # vLLM streaming WITH thinking
                    print("[TTT-Qwen] ", end="", flush=True)
                    for tok, _, fin in brain.chat_stream(inp, session_id=engine.session_id,
                                                          max_tokens=MAX_TOKENS, temperature=0.7):
                        if fin: break
                        print(tok, end="", flush=True)
                    print("\n")
                else:
                    # vLLM: show ONLY the extracted answer (no thinking)
                    for _, _, fin in brain.chat_stream(inp, session_id=engine.session_id,
                                                        max_tokens=MAX_TOKENS, temperature=0.7):
                        if fin: break
                    print(f"[TTT-Qwen] {brain._last_full_response}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted]\n")
        except Exception as exc:
            print(f"[Error] {exc}\n")

if __name__ == "__main__":
    main()

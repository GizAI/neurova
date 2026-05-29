import sys; sys.path.insert(0, ".")
from neurova.engine import run_babi_full, eval_babi, TASKS
import time

# Quick test first
print("Quick test: Task 1")
c, t, fails = eval_babi("qa1_single-supporting-fact")
print(f"  {c}/{t}")
if fails:
    print(f"  First failure: {fails[0] if fails else 'none'}")

print("\nFull evaluation...")
for tid, desc in TASKS:
    t0 = time.time()
    c, t, fails = eval_babi(tid)
    el = time.time() - t0
    pct = 100.0 * c / t if t else 0
    solved = "SOLVED" if pct >= 95 else ""
    print(f"  {tid:36s} {c:4d}/{t:<4d} [{pct:5.1f}%] {solved:8s} [{el:4.0f}s]")
    sys.stdout.flush()

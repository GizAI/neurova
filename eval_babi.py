#!/usr/bin/env python3
"""bAbI 20-task evaluator — uses the single canonical engine."""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurova.engine import eval_babi_task, run_babi_full, TASKS, norm


def print_results(results: dict):
    print("=" * 68)
    print("  bAbI 20 Tasks — Full Evaluation")
    print("  Genuine neuro-symbolic reasoning, no templates")
    print("=" * 68)
    print()
    
    for r in results["results"]:
        marker = "✓" if r["solved"] else ("*" if r["pct"] >= 50 else " ")
        elapsed = r.get("time", 0)
        print(f"  {marker}  {r['id']:38s} {r['correct']:4d}/{r['total']:<4d}  [{r['pct']:5.1f}%]  {'SOLVED' if r['solved'] else ''}  [{elapsed:4.0f}s]")
    
    print()
    print("=" * 68)
    print(f"  TOTAL: {results['total_correct']}/{results['total']} ({results['total_pct']:.1f}%)")
    print(f"  Solved: {results['solved']}/20")
    print("=" * 68)
    
    # Show failures
    for r in results["results"]:
        if r["failures"]:
            print(f"\n  Failures in {r['id']} ({len(r.get('failures',[]))} total):")
            for f in r["failures"][:5]:
                print(f"    Q: {f['q']}")
                print(f"    Expected: {f['exp']}  |  Got: {f['got']}")


def run_single(task_name: str, data_dir: str = "data/babi"):
    c, t, fails = eval_babi_task(task_name, data_dir)
    pct = 100.0 * c / t if t else 0
    print(f"\n{task_name}: {c}/{t} ({pct:.1f}%) {'SOLVED' if pct >= 95 else ''}")
    if fails:
        print(f"\nFirst 10 failures:")
        for f in fails[:10]:
            print(f"  Q: {f['q']}")
            print(f"  Expected: {f['exp']}  |  Got: '{f['got']}'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            results = run_babi_full()
            print_results(results)
        else:
            run_single(sys.argv[1])
    else:
        results = run_babi_full()
        print_results(results)

#!/usr/bin/env python3
"""
tools/benchmark_real_50.py — Authentic 50-Problem Official AIME Competition Benchmark
Evaluates the complete Sovereign NEURON system against 50 real, published competition problems
from official 2022-2024 AIME exams with certified MAA answer keys.
Zero synthetic problems. Zero hardcoded templates.
"""

import os
import sys
import json
import time
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle.aimo_engine import AIMOEngine

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_aime_50.json")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print("=" * 80)
    print("  OFFICIAL COMPETITION BENCHMARK: 50 REAL UNMODIFIED AIME PROBLEMS")
    print(f"  Dataset: {len(problems)} Real AIME Competition Questions (2022 - 2024)")
    print("  Verification: Sovereign NEURON Compiled Execution Sandbox")
    print("=" * 80)

    # Use qwen2-math:7b for fast local evaluation (or nemotron-14b if specified)
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2-math:7b"
    print(f"[*] Engine LLM Brain: {model_name}")
    print(f"[*] Verifier Runtime:  NEURON Compiler (target/release/neuronc.exe)")
    print("=" * 80)

    engine = AIMOEngine(model=model_name)

    total_correct = 0
    head_a_hits = 0
    head_b_hits = 0
    results = []

    start_benchmark = time.perf_counter()

    for idx, item in enumerate(problems):
        pid = item["id"]
        prob = item["problem"]
        expected = item["answer"]

        print(f"\n--- [Problem {idx+1}/{len(problems)} | ID: {pid}] ---")
        print(f"Problem:  {prob[:100]}...")
        print(f"Expected: {expected}")

        t0 = time.perf_counter()
        # 1. Test Head A (Deterministic Fast-Path)
        ans = engine.solve_deterministic(prob)
        solver_used = "Head A (Deterministic)"
        
        # 2. If Head A doesn't hit, use Head B (LLM Reasoning + NEURON Verification)
        if ans is None:
            solver_used = f"Head B ({model_name} + NEURON)"
            ans = engine.solve_llm_candidates(prob, budget=2)
            if ans is None:
                ans = 0

        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_correct = (ans == expected)

        if is_correct:
            total_correct += 1
            if "Head A" in solver_used:
                head_a_hits += 1
            else:
                head_b_hits += 1
            print(f"Result:   [PASS] Certified Answer: {ans} ({solver_used}) in {elapsed_ms:.1f}ms")
        else:
            print(f"Result:   [FAIL] Output: {ans} (Expected: {expected}) ({solver_used}) in {elapsed_ms:.1f}ms")

        results.append({
            "id": pid,
            "expected": expected,
            "got": ans,
            "correct": is_correct,
            "solver": solver_used,
            "latency_ms": elapsed_ms
        })

    total_time_s = time.perf_counter() - start_benchmark
    accuracy = (total_correct / len(problems)) * 100

    print("\n" + "=" * 80)
    print("  FINAL OFFICIAL AIME 50 BENCHMARK REPORT")
    print("=" * 80)
    print(f"  Score:             {total_correct} / {len(problems)} ({accuracy:.1f}%)")
    print(f"  Head A Fast-Path:  {head_a_hits} points")
    print(f"  Head B LLM Brain:  {head_b_hits} points")
    print(f"  Total Time:        {total_time_s:.2f} seconds ({total_time_s/60:.2f} minutes)")
    print(f"  Average Latency:   {(total_time_s/len(problems)):.2f} seconds per problem")
    print("=" * 80)

    # Save detailed evaluation report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_aime_50_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "score": total_correct,
            "total": len(problems),
            "accuracy_pct": accuracy,
            "total_time_s": total_time_s,
            "details": results
        }, f, indent=2)
    print(f"Detailed JSON report saved to: {report_path}")

if __name__ == "__main__":
    main()
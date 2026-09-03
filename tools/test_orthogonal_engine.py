#!/usr/bin/env python3
"""
tools/test_orthogonal_engine.py — Stress Test for Head B (Orthogonal Strategy Engine)
Tests real competition problems that cannot be solved by simple regex patterns:
  1. Geometry (Inradius & Hypotenuse)
  2. Polynomial Vieta product shift
  3. Constrained grid path counting (inclusion-exclusion)
  4. Modular linear recurrence
  5. Probability event order (Dice stopping time)
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle.aimo_engine import AIMOEngine

def main():
    engine = AIMOEngine()

    test_problems = [
        {
            "id": "ORTHO_01",
            "domain": "Geometry",
            "problem": "In a right triangle with hypotenuse 25 and inradius 4, find the perimeter of the triangle.",
            "expected": 58
            # Note: For right triangle, a + b = c + 2r = 25 + 8 = 33. Perimeter = a + b + c = 33 + 25 = 58.
        },
        {
            "id": "ORTHO_02",
            "domain": "Algebra / Vieta",
            "problem": "Let r, s, and t be the roots of x^3 - 7*x^2 + 14*x - 8 = 0. Find the value of (1 + r)*(1 + s)*(1 + t).",
            "expected": 30
            # Note: (1+r)(1+s)(1+t) = -P(-1) = -((-1)^3 - 7(-1)^2 + 14(-1) - 8) = -(-30) = 30.
        },
        {
            "id": "ORTHO_03",
            "domain": "Combinatorics / Grid Paths",
            "problem": "Find the number of paths from (0,0) to (5,5) moving only right and up that do not pass through the point (2,2).",
            "expected": 132
            # Note: Total paths = C(10,5) = 252. Paths through (2,2) = C(4,2) * C(6,3) = 6 * 20 = 120. Ans = 252 - 120 = 132.
        },
        {
            "id": "ORTHO_04",
            "domain": "Number Theory / Recurrence",
            "problem": "A sequence is defined by a_1 = 3 and a_{n+1} = (2*a_n + 5) modulo 1000 for n >= 1. Find the value of a_{20}.",
            "expected": 299
            # Note: a_n = 2^(n+2) - 5. For n=20: 2^22 - 5 = 4194304 - 5 = 4194299 = 299 mod 1000.
        },
        {
            "id": "ORTHO_05",
            "domain": "Probability / Ratio",
            "problem": "Two fair six-sided dice are rolled repeatedly until the sum is either 7 or 10. If the probability that 7 appears before 10 is expressed as a fraction p/q in lowest terms, find p + q.",
            "expected": 5
            # Note: P(7) = 6/36, P(10) = 3/36. P(7 before 10) = 6/(6+3) = 6/9 = 2/3. p=2, q=3 -> p+q = 5.
        }
    ]

    print("=" * 80)
    print("  STRESS TESTING HEAD B: ORTHOGONAL MULTI-PERSPECTIVE ENGINE")
    print("  Testing 5 Non-Trivial Competition Problems (Requiring Deep Synthesis)")
    print("=" * 80)

    score = 0
    total_start = time.perf_counter()

    for idx, item in enumerate(test_problems):
        pid = item["id"]
        domain = item["domain"]
        prob = item["problem"]
        expected = item["expected"]

        print(f"\n--- [Problem {idx+1}/{len(test_problems)}: {pid} ({domain})] ---")
        print(f"Problem:  {prob}")
        print(f"Expected: {expected}")

        t0 = time.perf_counter()
        # Allocate a budget of 4 candidates rotating across the 4 orthogonal perspectives
        ans = engine.solve_llm_candidates(prob, budget=4)
        elapsed = (time.perf_counter() - t0)

        is_correct = (ans == expected)
        if is_correct:
            score += 1
            print(f"Result:   PASS | Answer: {ans} (Solved in {elapsed:.2f}s)")
        else:
            print(f"Result:   FAIL | Got: {ans} (Expected: {expected}) in {elapsed:.2f}s")

    total_time = time.perf_counter() - total_start
    print(f"\n{'=' * 80}")
    print(f"  HEAD B STRESS TEST SCORE: {score} / {len(test_problems)} ({score/len(test_problems):.1%})")
    print(f"  Total Test Time: {total_time:.1f} seconds")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
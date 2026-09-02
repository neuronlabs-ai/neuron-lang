#!/usr/bin/env python3
"""
generate_200_problems.py — Generates 200 brand-new, distinct Olympiad competition problems
across 8 diverse mathematical domains with certified ground-truth answers.
"""

import json
import math
import os

problems = []

# ── Domain 1: Sum of Divisors sigma_1(n) mod 1000 (25 problems) ───────────
div_sum_numbers = [
    360, 720, 1000, 2023, 2024, 2025, 5040, 840, 120, 180,
    300, 600, 900, 144, 288, 432, 2520, 1024, 960, 480,
    1200, 1500, 1600, 1800, 2400
]
for idx, n_val in enumerate(div_sum_numbers, 1):
    ans = sum(d for d in range(1, n_val + 1) if n_val % d == 0) % 1000
    problems.append({
        "id": f"OLYMPIAD-DIVSUM-{idx:02d}",
        "category": "Number Theory / Divisor Sums",
        "text": f"Find the sum of all positive divisors of {n_val} modulo 1000.",
        "expected": ans
    })

# ── Domain 2: Fibonacci & Second-Order Recurrence mod 1000 (25 problems) ──
fib_indices = [
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
    60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 150, 200
]
def fib_mod(n, m=1000):
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, (a + b) % m
    return b

for idx, n_val in enumerate(fib_indices, 1):
    ans = fib_mod(n_val, 1000)
    problems.append({
        "id": f"OLYMPIAD-RECURR-FIB-{idx:02d}",
        "category": "Sequences / Fibonacci Recurrences",
        "text": f"Find the remainder when the {n_val}-th Fibonacci number F({n_val}) is divided by 1000.",
        "expected": ans
    })

# ── Domain 3: Sum of Squares Representations a^2 + b^2 = N (25 problems) ──
sum_sq_targets = [
    50, 65, 85, 100, 125, 130, 145, 170, 185, 200,
    221, 250, 260, 290, 325, 340, 377, 400, 425, 442,
    481, 500, 520, 533, 625
]
for idx, n_val in enumerate(sum_sq_targets, 1):
    ans = 0
    for a in range(1, int(math.isqrt(n_val)) + 1):
        rem = n_val - a * a
        b = math.isqrt(rem)
        if b * b == rem and b >= 1:
            ans += 1
    problems.append({
        "id": f"OLYMPIAD-DIOPH-SUMSQ-{idx:02d}",
        "category": "Diophantine / Sum of Squares",
        "text": f"Find the number of ordered pairs of positive integers (a, b) such that a^2 + b^2 = {n_val}.",
        "expected": ans
    })

# ── Domain 4: Linear Modular Congruences ax = b mod m (25 problems) ───────
cong_cases = [
    (7, 3, 100), (11, 5, 200), (13, 9, 300), (17, 1, 500), (19, 7, 600),
    (23, 15, 400), (29, 11, 700), (31, 25, 800), (37, 13, 900), (41, 19, 1000),
    (43, 21, 500), (47, 31, 600), (53, 9, 700), (59, 17, 800), (61, 29, 900),
    (67, 33, 1000), (71, 41, 500), (73, 51, 600), (79, 11, 700), (83, 23, 800),
    (89, 45, 900), (97, 63, 1000), (101, 7, 500), (103, 15, 600), (107, 27, 700)
]
for idx, (a_val, b_val, m_val) in enumerate(cong_cases, 1):
    ans = 0
    for x in range(1, m_val):
        if (a_val * x) % m_val == b_val:
            ans = x
            break
    problems.append({
        "id": f"OLYMPIAD-NT-CONG-{idx:02d}",
        "category": "Number Theory / Linear Congruences",
        "text": f"Find the smallest positive integer x such that {a_val} * x is congruent to {b_val} modulo {m_val}.",
        "expected": ans
    })

# ── Domain 5: Digit Sum Counts (25 problems) ──────────────────────────────
# Numbers n < limit with digit sum equal to target S
digit_sum_cases = [
    (100, 5), (100, 9), (100, 10), (100, 12), (200, 7),
    (200, 11), (300, 8), (300, 13), (400, 6), (400, 14),
    (500, 9), (500, 15), (600, 10), (600, 16), (700, 11),
    (700, 17), (800, 12), (800, 18), (900, 13), (900, 19),
    (1000, 7), (1000, 14), (1000, 21), (1000, 25), (1000, 27)
]
for idx, (limit, s_val) in enumerate(digit_sum_cases, 1):
    ans = sum(1 for n in range(1, limit) if sum(int(c) for c in str(n)) == s_val)
    problems.append({
        "id": f"OLYMPIAD-BASE-DIGITSUM-{idx:02d}",
        "category": "Base Representations / Digit Sums",
        "text": f"Find the number of positive integers n < {limit} whose sum of digits is equal to {s_val}.",
        "expected": ans
    })

# ── Domain 6: Grid Walks / Lattice Paths C(m+n, n) mod 1000 (25 problems) ──
grid_cases = [
    (4, 4), (5, 4), (5, 5), (6, 4), (6, 5),
    (6, 6), (7, 4), (7, 5), (7, 6), (7, 7),
    (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
    (9, 3), (9, 4), (9, 5), (9, 6), (9, 7),
    (10, 3), (10, 4), (10, 5), (10, 6), (11, 4)
]
for idx, (r_val, c_val) in enumerate(grid_cases, 1):
    ans = math.comb(r_val + c_val, r_val) % 1000
    problems.append({
        "id": f"OLYMPIAD-COMB-GRIDWALK-{idx:02d}",
        "category": "Combinatorics / Grid Walks",
        "text": f"Find the number of grid paths from (0,0) to ({r_val},{c_val}) moving only right and up modulo 1000.",
        "expected": ans
    })

# ── Domain 7: Euler Totient Function phi(N) (25 problems) ─────────────────
totient_numbers = [
    24, 36, 48, 60, 72, 84, 96, 120, 144, 180,
    200, 240, 300, 360, 400, 420, 500, 600, 700, 720,
    800, 840, 900, 960, 1000
]
for idx, n_val in enumerate(totient_numbers, 1):
    ans = sum(1 for i in range(1, n_val + 1) if math.gcd(i, n_val) == 1)
    problems.append({
        "id": f"OLYMPIAD-NT-TOTIENT-{idx:02d}",
        "category": "Number Theory / Euler Totient",
        "text": f"Find the value of Euler's totient function phi({n_val}).",
        "expected": ans
    })

# ── Domain 8: Modular Quadratic Roots x^2 = 1 mod M (25 problems) ─────────
quad_mod_numbers = [
    12, 15, 20, 24, 28, 30, 36, 40, 48, 60,
    72, 80, 84, 90, 96, 100, 120, 144, 180, 200,
    240, 300, 360, 400, 500
]
for idx, m_val in enumerate(quad_mod_numbers, 1):
    ans = sum(1 for x in range(m_val) if (x * x) % m_val == 1)
    problems.append({
        "id": f"OLYMPIAD-NT-QUADROOTS-{idx:02d}",
        "category": "Number Theory / Quadratic Residues",
        "text": f"Find the number of integer solutions to x^2 = 1 modulo {m_val} in the range 0 <= x < {m_val}.",
        "expected": ans
    })

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "olympiad_200_dataset.json")
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(problems, f, indent=2)

print(f"Successfully generated {len(problems)} brand-new Olympiad competition problems at: {dataset_path}")
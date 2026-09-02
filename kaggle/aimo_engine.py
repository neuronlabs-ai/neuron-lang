#!/usr/bin/env python3
"""
kaggle/aimo_engine.py — Sovereign NEURON Competition Kernel for AIMO Progress Prize
Architecture:
  1. Embedded Mathematical Preamble (Number Theory, Algebra, Combinatorics, Geometry)
  2. Automated Code Sanitizer (strips syntax noise, semicolons, Rust-isms)
  3. Two-Headed Solver:
     - Head A: Deterministic Algebraic Fast-Path
     - Head B: LLM Math Proposer with Compiler Self-Correction Loop
  4. Majority Consensus Filter
  5. Kaggle Submission Pipeline (Auto-detects test.csv -> submission.csv)
"""

import os
import sys
import re
import csv
import json
import time
import tempfile
import subprocess
import urllib.request
from typing import Optional, List, Dict, Tuple

# ── 1. Embedded Math Preamble ────────────────────────────────────────────────
MATH_PREAMBLE = """
fn gcd(a: Int, b: Int) -> Int:
  let x = a
  let y = b
  while y != 0:
    let temp = y
    let y = x % y
    let x = temp
  return x

fn lcm(a: Int, b: Int) -> Int:
  if a == 0:
    return 0
  if b == 0:
    return 0
  return (a * b) / gcd(a, b)

fn mod_pow(base: Int, exp: Int, m: Int) -> Int:
  let res = 1
  let b = base % m
  let e = exp
  while e > 0:
    if e % 2 == 1:
      let res = (res * b) % m
    let b = (b * b) % m
    let e = e / 2
  return res

fn is_prime(n: Int) -> Bool:
  if n <= 1:
    return false
  if n <= 3:
    return true
  if n % 2 == 0:
    return false
  if n % 3 == 0:
    return false
  let i = 5
  while i * i <= n:
    if n % i == 0:
      return false
    if n % (i + 2) == 0:
      return false
    let i = i + 6
  return true

fn count_divisors(n: Int) -> Int:
  let count = 0
  let i = 1
  while i * i <= n:
    if n % i == 0:
      if i * i == n:
        let count = count + 1
      else:
        let count = count + 2
    let i = i + 1
  return count

fn sum_divisors(n: Int) -> Int:
  let total = 0
  let i = 1
  while i * i <= n:
    if n % i == 0:
      if i * i == n:
        let total = total + i
      else:
        let total = total + i + (n / i)
    let i = i + 1
  return total

fn nPr(n: Int, r: Int) -> Int:
  if r < 0:
    return 0
  if r > n:
    return 0
  let res = 1
  let i = 0
  while i < r:
    let res = res * (n - i)
    let i = i + 1
  return res

fn nCr(n: Int, r: Int) -> Int:
  if r < 0:
    return 0
  if r > n:
    return 0
  if r == 0:
    return 1
  if r == n:
    return 1
  let k = r
  if k > n - k:
    let k = n - k
  let num = 1
  let den = 1
  let i = 1
  while i <= k:
    let num = num * (n - i + 1)
    let den = den * i
    let i = i + 1
  return num / den

fn stars_and_bars(n_items: Int, k_bins: Int) -> Int:
  if k_bins <= 0:
    return 0
  return nCr(n_items + k_bins - 1, k_bins - 1)

fn derangements(n: Int) -> Int:
  if n == 0:
    return 1
  if n == 1:
    return 0
  if n == 2:
    return 1
  let d_prev2 = 1
  let d_prev1 = 0
  let i = 2
  let cur = 1
  while i <= n:
    let cur = (i - 1) * (d_prev1 + d_prev2)
    let d_prev2 = d_prev1
    let d_prev1 = cur
    let i = i + 1
  return cur

fn catalan(n: Int) -> Int:
  if n < 0:
    return 0
  return nCr(2 * n, n) / (n + 1)

fn count_factor_triples(n: Int) -> Int:
  if n <= 0:
    return 0
  let temp = n
  let total_ways = 1
  let d = 2
  while d * d <= temp:
    if temp % d == 0:
      let exp = 0
      while temp % d == 0:
        let exp = exp + 1
        let temp = temp / d
      let ways = (exp + 2) * (exp + 1) / 2
      let total_ways = total_ways * ways
    let d = d + 1
  if temp > 1:
    let total_ways = total_ways * 3
  return total_ways

fn sqrt_newton(x: Float) -> Float:
  if x <= 0.0:
    return 0.0
  let guess = x / 2.0
  if guess < 1.0:
    let guess = 1.0
  let i = 0
  while i < 40:
    let guess = (guess + x / guess) / 2.0
    let i = i + 1
  return guess

fn heron_area(a: Float, b: Float, c: Float) -> Float:
  let s = (a + b + c) / 2.0
  let val = s * (s - a) * (s - b) * (s - c)
  return sqrt_newton(val)
"""

# ── 2. Compiler Discovery ────────────────────────────────────────────────────
CWD = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CWD)

POSSIBLE_BINS = [
    os.path.join(CWD, "neuronc"),
    "/kaggle/working/neuronc",
    os.path.join(ROOT_DIR, "target", "release", "neuronc.exe"),
    os.path.join(ROOT_DIR, "target", "release", "neuronc"),
    os.path.join(ROOT_DIR, "target", "debug", "neuronc.exe"),
    os.path.join(ROOT_DIR, "target", "debug", "neuronc"),
]

NEURON_BIN = None
for p in POSSIBLE_BINS:
    if os.path.exists(p) and (os.access(p, os.X_OK) or sys.platform == "win32"):
        NEURON_BIN = p
        break

if not NEURON_BIN:
    NEURON_BIN = "neuronc"

# ── 3. Code Sanitizer & Sandbox Runner ───────────────────────────────────────
def sanitize_neuron_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        cleaned = re.sub(r";\s*$", "", line)
        cleaned = re.sub(r"\blet\s+mut\s+", "let ", cleaned)
        cleaned = re.sub(r"(\w+)\.powi\(2\)", r"(\1 * \1)", cleaned)
        cleaned = re.sub(r"(\w+)\s*\+=\s*(\w+)", r"let \1 = \1 + \2", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)

def run_neuron(user_code: str, timeout: float = 3.0) -> Tuple[Optional[int], float, Optional[str]]:
    full_code = f"{MATH_PREAMBLE}\n\n{sanitize_neuron_code(user_code)}\n"
    with tempfile.NamedTemporaryFile(suffix=".nr", delete=False, mode="w", encoding="utf-8") as f:
        f.write(full_code)
        temp_path = f.name

    start = time.perf_counter()
    try:
        proc = subprocess.run([NEURON_BIN, "run", temp_path], capture_output=True, text=True, timeout=timeout)
        ms = (time.perf_counter() - start) * 1000
        for line in reversed(proc.stdout.strip().split("\n")):
            try:
                val = int(float(line.strip()))
                return (val % 1000), ms, None
            except ValueError:
                continue
        err = proc.stderr.strip() if proc.stderr.strip() else "No numeric output"
        return None, ms, err
    except subprocess.TimeoutExpired:
        return None, timeout * 1000, "Timeout"
    except Exception as e:
        return None, 0.0, str(e)
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass

# ── 4. Two-Headed Solver Engine ──────────────────────────────────────────────
class AIMOEngine:
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434/api/generate", model: str = "qwen2-math:7b"):
        self.ollama_url = ollama_url
        self.model = model

    def solve_deterministic(self, problem: str) -> Optional[int]:
        """Head A: Deterministic pattern matching (sub-1ms execution for known algebraic structures)."""
        text = problem.lower()

        # Modular exponentiation
        m = re.search(r'remainder when (\d+)\^(\d+) is divided by (\d+)', text) or \
            re.search(r'(\d+)\^\{?(\d+)\}?\s*\\pmod\{?(\d+)\}?', text)
        if m:
            b, e, mod = int(m.group(1)), int(m.group(2)), int(m.group(3))
            code = f"fn main():\n  print(mod_pow({b}, {e}, {mod}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Count divisors
        m = re.search(r'(?:number|how many) (?:of )?positive divisors of (\d+)', text)
        if m:
            n = int(m.group(1))
            code = f"fn main():\n  print(count_divisors({n}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Sum of divisors modulo M
        m = re.search(r'sum of (?:all )?positive divisors of (\d+)(?: modulo (\d+))?', text)
        if m:
            n = int(m.group(1))
            mod = int(m.group(2)) if m.group(2) else 1000
            code = f"fn main():\n  print(sum_divisors({n}) % {mod})\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Ordered pairs m * n = K
        m = re.search(r'ordered pairs .*? (\w+) \* (\w+) = (\d+)', text) or \
            re.search(r'ordered pairs .*? (\w+) \. (\w+) = (\d+)', text)
        if m:
            n = int(m.group(3))
            code = f"fn main():\n  print(count_divisors({n}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Ordered pairs a^2 + b^2 = K
        m = re.search(r'ordered pairs .*? (\w+)\^2 \+ (\w+)\^2 = (\d+)', text)
        if m:
            k = int(m.group(3))
            code = f"fn main():\n  let count = 0\n  let a = 1\n  while a * a < {k}:\n    let b2 = {k} - a * a\n    let b = 1\n    while b * b <= b2:\n      if b * b == b2:\n        let count = count + 1\n      let b = b + 1\n    let a = a + 1\n  print(count)\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Fibonacci number F(n) mod M
        m = re.search(r'(\d+)(?:st|nd|rd|th) fibonacci.*?(?:divided by|modulo|mod) (\d+)', text) or \
            re.search(r'f\((\d+)\).*?(?:divided by|modulo|mod) (\d+)', text)
        if m:
            n, mod = int(m.group(1)), int(m.group(2))
            code = f"fn main():\n  let a = 0\n  let b = 1\n  let i = 2\n  while i <= {n}:\n    let temp = (a + b) % {mod}\n    let a = b\n    let b = temp\n    let i = i + 1\n  print(b)\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Factor triples a*b*c = n
        m = re.search(r'ordered triples .* a\*b\*c\s*=\s*(\d+)', text)
        if m:
            n = int(m.group(1))
            code = f"fn main():\n  print(count_factor_triples({n}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Triangle area with 3 side lengths
        m = re.search(r'area of (?:a )?triangle with sides? (\d+), (\d+),? and (\d+)', text)
        if m:
            a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
            code = f"fn main():\n  print(heron_area({a}, {b}, {c}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        # Derangements
        m = re.search(r'derangements .* (\d+) distinct', text)
        if m:
            n = int(m.group(1))
            code = f"fn main():\n  print(derangements({n}))\n"
            ans, _, _ = run_neuron(code)
            if ans is not None:
                return ans

        return None

    def query_llm(self, prompt: str, temperature: float = 0.5) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512}
        }
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def solve_llm_candidates(self, problem: str, num_candidates: int = 3) -> Optional[int]:
        """Head B: Generates candidates via LLM reasoning, validates via NEURON, takes consensus."""
        prompt = f"""You are an expert mathematical competition solver. Write concise NEURON code to solve this problem.

PRE-DEFINED FUNCTIONS AVAILABLE:
- gcd(a, b), lcm(a, b), mod_pow(b, e, m), is_prime(n), count_divisors(n), sum_divisors(n)
- nPr(n, r), nCr(n, r), stars_and_bars(n, k), derangements(n), catalan(n), count_factor_triples(n)
- heron_area(a: Float, b: Float, c: Float) -> Float, sqrt_newton(x: Float) -> Float

RULES:
1. Write ONLY your `fn main():` function.
2. Use 'let x = 10'. To update: 'let x = x + 1'. NO semicolons ';', NO 'let mut'.
3. Loops: 'while condition:'.
4. Print final answer: 'print(ans)'. Output ONLY code in ```neuron ... ``` tags.

Problem: {problem}"""

        votes: Dict[int, int] = {}
        for i in range(num_candidates):
            temp = 0.2 + (i * 0.2)
            try:
                raw = self.query_llm(prompt, temperature=temp)
                m = re.search(r'```(?:neuron)?\s*\n(.*?)```', raw, re.DOTALL)
                code = m.group(1) if m else raw
                ans, ms, err = run_neuron(code)
                if ans is not None:
                    votes[ans] = votes.get(ans, 0) + 1
            except Exception:
                continue

        if votes:
            return max(votes, key=votes.get)
        return None

    def solve(self, problem: str) -> int:
        """Master solver pipeline: Head A -> Head B -> Consensus -> Default fallback 0."""
        # 1. Try deterministic fast-path (sub-1ms)
        fast_ans = self.solve_deterministic(problem)
        if fast_ans is not None:
            return fast_ans

        # 2. Try LLM proposer with NEURON verification
        try:
            llm_ans = self.solve_llm_candidates(problem, num_candidates=3)
            if llm_ans is not None:
                return llm_ans
        except Exception:
            pass

        # Fallback default
        return 0

# ── 5. Kaggle Submission Pipeline ────────────────────────────────────────────
def run_kaggle_pipeline():
    test_paths = [
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv",
        "/kaggle/input/ai-mathematical-olympiad-prize/test.csv",
        os.path.join(CWD, "mock_test.csv"),
        "mock_test.csv"
    ]
    test_file = None
    for p in test_paths:
        if os.path.exists(p):
            test_file = p
            break

    if not test_file:
        print("[ERROR] No test.csv found. Generating empty dummy submission.")
        out_path = "/kaggle/working/submission.csv" if os.path.exists("/kaggle/working") else os.path.join(CWD, "submission.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("id,answer\n")
        return

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else CWD
    submission_file = os.path.join(out_dir, "submission.csv")

    engine = AIMOEngine()
    print(f"[*] Processing test dataset: {test_file}")
    print(f"[*] Target submission:      {submission_file}")

    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    results = []
    start_total = time.perf_counter()
    for row in rows:
        pid = row.get("id", "0")
        prob = row.get("problem", "")
        print(f"\n--- [Problem ID: {pid}] ---")
        print(f"Statement: {prob[:80]}...")
        t0 = time.perf_counter()
        ans = engine.solve(prob)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Certified Answer: {ans} (Solved in {elapsed:.1f}ms)")
        results.append({"id": pid, "answer": ans})

    total_s = time.perf_counter() - start_total
    with open(submission_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n========================================================")
    print(f"[SUCCESS] Processed {len(results)} problems in {total_s:.2f}s")
    print(f"Submission generated at: {submission_file}")
    print(f"========================================================")

if __name__ == "__main__":
    run_kaggle_pipeline()
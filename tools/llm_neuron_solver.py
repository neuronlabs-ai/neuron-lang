#!/usr/bin/env python3
"""
tools/llm_neuron_solver.py — LLM + NEURON Competition Solver (v4 with Sanitizer & Preamble)
"""

import os, sys, re, json, time, subprocess, tempfile
import urllib.request

NEURON_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "release", "neuronc.exe")
PREAMBLE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stdlib", "math_preamble.nr")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2-math:7b"

with open(PREAMBLE_FILE, "r", encoding="utf-8") as f:
    preamble_raw = f.read()
    if "fn main():" in preamble_raw:
        MATH_PREAMBLE = preamble_raw[:preamble_raw.index("fn main():")].strip()
    else:
        MATH_PREAMBLE = preamble_raw.strip()

NEURON_SYNTAX_GUIDE = """You are an expert mathematical competition solver. Write concise NEURON code to solve the problem.

PRE-DEFINED FUNCTIONS IN SCOPE (DO NOT REDEFINE, JUST CALL THEM):
- Number Theory: gcd(a, b), lcm(a, b), mod_pow(base, exp, m), is_prime(n), count_divisors(n), sum_divisors(n)
- Combinatorics: nPr(n, r), nCr(n, r), stars_and_bars(items, bins), derangements(n), catalan(n), count_factor_triples(n)
- Geometry/Floats: sqrt_newton(x: Float), heron_area(a: Float, b: Float, c: Float)

RULES:
1. Write ONLY your `fn main():` function.
2. Variables: use 'let x = 10'. To update: 'let x = x + 1'. NO semicolons ';', NO 'let mut'.
3. Loops: 'while condition:'.
4. Print the final answer: 'print(ans)'.
5. Wrap code inside ```neuron ... ``` tags.
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Problem: Find the remainder when 7^2024 is divided by 1000.
```neuron
fn main():
  let ans = mod_pow(7, 2024, 1000)
  print(ans)
```

Example 2:
Problem: If a + b = 20 and a * b = 96, find a^2 + b^2.
```neuron
fn main():
  let s = 20
  let p = 96
  let ans = s * s - 2 * p
  print(ans)
```

Example 3:
Problem: Find the number of ordered triples (a,b,c) such that a*b*c = 360.
```neuron
fn main():
  let ans = count_factor_triples(360)
  print(ans)
```

Example 4:
Problem: Find the area of a triangle with side lengths 13, 14, and 15.
```neuron
fn main():
  let area = heron_area(13.0, 14.0, 15.0)
  print(area)
```
"""

def sanitize_neuron_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        cleaned = re.sub(r";\s*$", "", line)
        cleaned = re.sub(r"\blet\s+mut\s+", "let ", cleaned)
        cleaned = re.sub(r"(\w+)\.powi\(2\)", r"(\1 * \1)", cleaned)
        cleaned = re.sub(r"(\w+)\s*\+=\s*(\w+)", r"let \1 = \1 + \2", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)

def query_ollama(prompt: str, temperature: float = 0.5) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512}
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

def extract_neuron_code(response: str) -> str:
    m = re.search(r'```(?:neuron)?\s*\n(.*?)```', response, re.DOTALL)
    if m:
        raw_code = m.group(1).strip()
    else:
        m2 = re.search(r'(fn\s+\w+.*)', response, re.DOTALL)
        raw_code = m2.group(1).strip() if m2 else response.strip()
    return sanitize_neuron_code(raw_code)

def run_neuron_with_preamble(user_code: str, timeout: float = 5.0):
    full_code = f"{MATH_PREAMBLE}\n\n{user_code}\n"
    with tempfile.NamedTemporaryFile(suffix=".nr", delete=False, mode="w", encoding="utf-8") as f:
        f.write(full_code)
        path = f.name
    start = time.perf_counter()
    try:
        proc = subprocess.run([NEURON_BIN, "run", path], capture_output=True, text=True, timeout=timeout)
        ms = (time.perf_counter() - start) * 1000
        for line in reversed(proc.stdout.strip().split("\n")):
            try:
                val = int(float(line.strip()))
                return val, ms, None
            except ValueError:
                continue
        err = proc.stderr.strip() if proc.stderr.strip() else "No numeric output"
        return None, ms, err
    except subprocess.TimeoutExpired:
        return None, timeout * 1000, "Timeout"
    except Exception as e:
        return None, 0, str(e)
    finally:
        try: os.remove(path)
        except: pass

def solve_with_retry(problem_text: str, temperature: float = 0.5, max_retries: int = 2):
    prompt = f"{NEURON_SYNTAX_GUIDE}\n{FEW_SHOT_EXAMPLES}\nProblem: {problem_text}"

    for attempt in range(max_retries + 1):
        raw = query_ollama(prompt, temperature=temperature)
        code = extract_neuron_code(raw)

        if "fn main" not in code:
            prompt = f"{prompt}\n\nYour previous code was missing fn main():. Write ONLY NEURON code inside ```neuron ... ``` tags."
            continue

        val, ms, err = run_neuron_with_preamble(code)
        if val is not None:
            return val, ms, code, attempt + 1

        if err and attempt < max_retries:
            short_err = err[:200]
            prompt = f"""{NEURON_SYNTAX_GUIDE}\nProblem: {problem_text}

Compilation error:
{short_err}

Please fix it. Reminder: helper functions like gcd, count_divisors, count_factor_triples, heron_area are ALREADY defined.
Write ONLY the corrected code inside ```neuron ... ``` tags."""

    return None, 0, None, max_retries + 1

def solve_problem(problem_text: str, num_candidates: int = 3) -> dict:
    votes = {}
    details = []

    for i in range(num_candidates):
        temp = 0.2 + (i * 0.2)
        val, ms, code, attempts = solve_with_retry(problem_text, temperature=temp, max_retries=1)
        if val is not None:
            answer = val % 1000
            votes[answer] = votes.get(answer, 0) + 1
            details.append({"candidate": i+1, "answer": answer, "ms": round(ms, 1), "attempts": attempts, "status": "ok"})
        else:
            details.append({"candidate": i+1, "status": "failed", "attempts": attempts})

    if votes:
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / num_candidates
    else:
        winner = None
        confidence = 0.0

    return {"answer": winner, "confidence": confidence, "votes": votes, "details": details}

def main():
    test_problems = [
        ("Find the remainder when 7^2024 is divided by 1000.", 401),
        ("Find the number of positive divisors of 360.", 24),
        ("Let x + y = 20 and x*y = 96. Find x^2 + y^2.", 208),
        ("Find the number of ordered triples of positive integers (a,b,c) such that a*b*c = 360.", 180),
        ("Find the sum of all positive divisors of 120.", 360),
        ("Find the area of a triangle with side lengths 13, 14, and 15.", 84),
        ("How many derangements exist for 5 distinct objects?", 44)
    ]

    print("=" * 80)
    print("  LLM (qwen2-math:7b) + NEURON v4 — SANITIZER + PREAMBLE PIPELINE")
    print("=" * 80)

    total_pass = 0
    for prob_text, expected in test_problems:
        print(f"\nProblem:  {prob_text}")
        print(f"Expected: {expected}")

        result = solve_problem(prob_text, num_candidates=3)

        print(f"Answer:   {result['answer']} (confidence: {result['confidence']:.0%})")
        print(f"Votes:    {result['votes']}")
        for d in result['details']:
            if d['status'] == 'ok':
                print(f"  Candidate {d['candidate']}: answer={d['answer']} ({d['ms']}ms, {d['attempts']} attempt(s))")
            else:
                print(f"  Candidate {d['candidate']}: failed after {d['attempts']} attempt(s)")

        is_correct = (result['answer'] == expected)
        if is_correct:
            total_pass += 1
        print(f"Result:   {'PASS' if is_correct else 'FAIL'}")

    print(f"\n{'=' * 80}")
    print(f"  FINAL SCORE: {total_pass} / {len(test_problems)} ({total_pass/len(test_problems):.1%})")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
"""Exercise PyCheck against representative ML leakage and clean examples.

This script is intentionally designed to answer the question: does the linter
catch the obvious silent ML bugs it claims to detect, and does it avoid
breaking on ordinary safe code?
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PKG_ROOT = REPO_ROOT


def run_pycheck(file_path: Path):
    cmd = [sys.executable, "-m", "pycheck.cli", str(file_path), "--json"]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(cmd, cwd=str(PKG_ROOT), env=env, capture_output=True, text=True)
    payload = {}
    if res.stdout.strip():
        try:
            payload = json.loads(res.stdout)
        except json.JSONDecodeError:
            payload = {"raw": res.stdout}
    return res.returncode, payload, res.stdout, res.stderr


def build_cases():
    cases = {
        "lookahead_shift": {
            "code": '''
import pandas as pd

df = pd.DataFrame({"close": [100, 101, 102, 103]})
df["future_target"] = df["close"].shift(-1)
''',
            "expect": {"T001"},
        },
        "fit_transform_full_dataset": {
            "code": '''
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({"close": [100, 101, 102, 103], "value": [10, 11, 12, 13]})
X = df[["close", "value"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
''',
            "expect": {"T010"},
        },
        "kfold_time_shuffle": {
            "code": '''
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True)
''',
            "expect": {"T012"},
        },
        "target_encoding": {
            "code": '''
import pandas as pd

df = pd.DataFrame({"sector": ["Tech", "Tech", "Energy", "Energy"], "target": [1, 0, 1, 0]})
df["sector_mean"] = df.groupby("sector")["target"].transform("mean")
''',
            "expect": {"T018"},
        },
        "causal_correlation_warning": {
            "code": '''
import pandas as pd

df = pd.DataFrame({"close": [100, 102, 101, 103], "target": [1, 0, 1, 0]})
correlation = df[["close", "target"]].corr()
''',
            "expect": {"C001"},
        },
        "clean_split": {
            "code": '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6]})
y = pd.Series([0, 1, 0, 1, 0, 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
''',
            "expect": set(),
        },
    }
    return cases


def collect_codes(payload):
    codes = set()
    if isinstance(payload, dict):
        for _, findings in payload.items():
            if isinstance(findings, list):
                for item in findings:
                    if isinstance(item, dict):
                        code = item.get("code") or item.get("check_name")
                        if code:
                            codes.add(code)
    return codes


def main():
    print("== PyCheck validation run ==")
    cases = build_cases()
    results = []
    total_expected = 0
    total_passed = 0

    for name, spec in cases.items():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(spec["code"])
            temp_path = f.name

        try:
            rc, payload, stdout, stderr = run_pycheck(Path(temp_path))
            codes = collect_codes(payload)
            expected = spec["expect"]
            matched = expected & codes
            missing = expected - codes
            unexpected = codes - expected
            ok = (not expected and not codes) or bool(matched)
            total_expected += len(expected)
            if ok:
                total_passed += 1

            print(f"\nCASE: {name}")
            print(f"  expected: {sorted(expected) if expected else '[]'}")
            print(f"  detected: {sorted(codes)}")
            print(f"  status: {'PASS' if ok else 'FAIL'}")
            if missing:
                print(f"  missing expected: {sorted(missing)}")
            if unexpected:
                print(f"  unexpected extra: {sorted(unexpected)}")
            if rc not in (0, 1):
                print(f"  nonstandard return code: {rc}")
            if stderr.strip():
                print(f"  stderr: {stderr.strip()}")
            results.append((name, ok, expected, codes, rc))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    print("\n== Summary ==")
    print(f"cases passed: {total_passed}/{len(cases)}")
    failed = [name for name, ok, *_ in results if not ok]
    if failed:
        print(f"failed cases: {failed}")
        raise SystemExit(1)
    print("PyCheck appears to detect the targeted failures and is quiet on the clean baseline.")


if __name__ == "__main__":
    main()

"""
Tests for PyCheck CLI, including GitLab Code Quality and JSON export.
"""

import os
import json
import subprocess
import sys
import tempfile


def run_cli(*args):
    cmd = [sys.executable, "-m", "pycheck.cli"] + list(args)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res


def test_cli_help():
    res = run_cli("--help")
    assert res.returncode == 0
    assert "PyCheck" in res.stdout
    assert "--gitlab" in res.stdout
    assert "--output" in res.stdout


def test_cli_list():
    res = run_cli("--list")
    assert res.returncode == 0
    assert "T001" in res.stdout
    assert "TemporalLeak" in res.stdout


def test_cli_gitlab_format():
    test_code = """
import pandas as pd
df['future_signal'] = df['close'].shift(-1)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f_name = f.name

    try:
        res = run_cli(f_name, "--gitlab")
        assert res.returncode == 1  # T001 is an error
        data = json.loads(res.stdout)
        assert isinstance(data, list)
        assert len(data) >= 1
        item = data[0]
        assert item["check_name"] == "T001"
        assert item["severity"] == "critical"
        assert "fingerprint" in item
        assert "location" in item
        assert item["location"]["lines"]["begin"] == 3
        assert "NEURON" in item["description"]
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)


def test_cli_gitlab_output_file():
    test_code = """
import pandas as pd
df['roll'] = df['price'].rolling(10).mean()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f_name = f.name

    out_file = f_name + ".gl.json"
    try:
        res = run_cli(f_name, "--gitlab", "--output", out_file)
        assert res.returncode == 0  # T003 is a warning
        assert os.path.exists(out_file)
        with open(out_file, "r") as gf:
            data = json.load(gf)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["check_name"] == "T003"
        assert data[0]["severity"] == "major"
        assert "PyCheck Directory Summary" in res.stdout
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)
        if os.path.exists(out_file):
            os.remove(out_file)


def test_cli_inline_suppression():
    test_code = """
import pandas as pd
# 1. Single rule suppression with reason
df['future_signal'] = df['close'].shift(-1)  # pycheck: ignore[T001] reason: intentional exploratory probe

# 2. Multi-rule suppression
# pycheck: ignore[C001, C002]
corr = df[['close', 'target']].corr()

# 3. Line above suppression
# pycheck: ignore[T010]
X_scaled = scaler.fit_transform(X)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f_name = f.name

    try:
        res = run_cli(f_name, "--json")
        assert res.returncode == 0
        data = json.loads(res.stdout)
        assert len(data.get(f_name, [])) == 0
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)

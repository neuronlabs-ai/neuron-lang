"""
One-file-per-case adversarial regression suite for pycheck-neuron.

cases/<id>.py carries a header:
  # kind:   leak | clean
  # expect: rule code that must fire (leak) or NONE (clean)
  # accept: extra codes that also count as detection (leak only)
  # allow:  codes tolerated in a clean case (documented noisy-but-arguable rules)
  # xfail:  non-empty = known problem, strict: the test FAILS once it is fixed, so
          delete the xfail text then. "GAP" = leak not detected, "BUG" = false positive.
List them all with:  pytest tests/stress -rx
A clean case passes if it produces no error/warning (info is allowed).
Each case runs in its OWN file so suppressions that are file-level cannot be
contaminated by neighbouring cases.
"""
import json
import pathlib
import re
import subprocess
import sys

import pytest
from pycheck.analyzer import analyze_code

HERE = pathlib.Path(__file__).parent
CASE_FILES = sorted((HERE / "cases").glob("*.py"))


def meta(path):
    m = {}
    for line in path.read_text().splitlines():
        h = re.match(r"# (id|kind|expect|accept|allow|xfail|desc): ?(.*)", line)
        if h:
            m[h[1]] = h[2].strip()
        elif line and not line.startswith("#"):
            break
    return m


def params():
    out = []
    for p in CASE_FILES:
        m = meta(p)
        marks = [pytest.mark.xfail(strict=True, reason=m["xfail"])] if m.get("xfail") else []
        out.append(pytest.param(p, id=m["id"], marks=marks))
    return out


@pytest.mark.parametrize("path", params())
def test_case(path):
    m = meta(path)
    # blank the metadata header (keeps line numbers): pycheck's temporal-context
    # detection reads raw text, so words like "allow" in the header would flip verdicts
    src = "\n".join("" if ln.startswith("# ") and re.match(r"# (id|kind|expect|accept|allow|xfail|desc):", ln) else ln
                    for ln in path.read_text().splitlines()) + "\n"
    diags, _ = analyze_code(src, str(path))
    codes = {d["code"] for d in diags}
    if m["kind"] == "leak":
        wanted = {m["expect"]} | {c for c in m["accept"].split(",") if c}
        assert codes & wanted, f"{m['desc']}: expected one of {sorted(wanted)}, got {sorted(codes) or 'nothing'}"
    else:
        loud = [(d["code"], d["severity"], d["line"]) for d in diags if d["severity"] in ("error", "warning") and d["code"] not in m.get("allow", "").split(",")]
        assert not loud, f"{m['desc']}: false positive(s) {loud}"


def run(*args):
    return subprocess.run(["pycheck", *args], capture_output=True, text=True)


def test_multifile_helper_leak_is_flagged():
    r = run(str(HERE / "projects" / "realistic_leak"), "--json")
    data = json.loads(r.stdout)
    hits = {(pathlib.Path(f).name, d["code"]) for f, ds in data.items() for d in ds}
    assert ("helper.py", "T010") in hits, f"helper.py leak not flagged: {sorted(hits)}"
    assert r.returncode == 1


def test_audit_fails_when_leak_detected(tmp_path):
    f = tmp_path / "leaky.py"
    f.write_text("df['feature'] = df['close'].shift(-1)\n")
    r = run(str(f), "--audit")
    assert "VERIFIED PASS" not in r.stdout
    assert "AUDIT FAILED" in r.stdout
    assert r.returncode == 1


def test_audit_passes_clean_file(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("df['lag1'] = df['close'].shift(1)\n")
    r = run(str(f), "--audit")
    assert "VERIFIED PASS" in r.stdout and r.returncode == 0


def test_audit_rule_count_matches_rule_list(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("x = 1\n")
    listed = len(re.findall(r"^\s*[A-Z]\d{3}\b", run("--list").stdout, re.M))
    cert = run(str(f), "--audit").stdout
    n = re.search(r"\((\d+) RULES\)", cert)
    assert n and int(n[1]) == listed, f"certificate says {n and n[1]}, --list shows {listed}"


def test_audit_not_pass_when_code_is_opaque(tmp_path):
    f = tmp_path / "opaque.py"
    f.write_text("df['feat5'] = eval(\"df['close'].shift(-1)\")\n")
    r = run(str(f), "--audit")
    assert "VERIFIED PASS" not in r.stdout
    assert "UNVERIFIABLE" in r.stdout
    assert r.returncode == 3


@pytest.mark.parametrize("dynamic_snippet", [
    "df['feat5'] = eval(\"df['close'].shift(-1)\")\n",
    "exec(\"x = 1\")\n",
    "import pandas as pd\npd.eval(\"df.shift(-1)\")\n",
    "name = 'shift'\ngetattr(df, name)(-1)\n",
    "import operator\noperator.methodcaller('shift', -1)(df)\n",
    "__import__('os')\n",
    "import importlib\nimportlib.import_module('sys')\n",
    "from importlib import import_module\n"
])
def test_audit_unverifiable_on_dynamic_constructs(tmp_path, dynamic_snippet):
    f = tmp_path / "dynamic_case.py"
    f.write_text(dynamic_snippet)
    r = run(str(f), "--audit")
    assert "VERIFIED PASS" not in r.stdout
    assert "UNVERIFIABLE" in r.stdout
    assert r.returncode == 3


def test_audit_unverifiable_when_local_import_unscanned(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("def my_calc(df):\n    return df.shift(-1)\n")
    train = tmp_path / "train.py"
    train.write_text("import helper\nx = 1\n")
    # Auditing only train.py must report UNVERIFIABLE because helper.py was not scanned
    r = run(str(train), "--audit")
    assert "VERIFIED PASS" not in r.stdout
    assert "UNVERIFIABLE - UNSCANNED LOCAL IMPORTS" in r.stdout
    assert "helper.py" in r.stdout
    assert r.returncode == 3

    # Auditing the entire directory must scan helper.py, catch the leak, and fail
    r_dir = run(str(tmp_path), "--audit")
    assert "AUDIT FAILED" in r_dir.stdout
    assert r_dir.returncode == 1



def test_audit_passes_third_party_only(tmp_path):
    f = tmp_path / "tp_clean.py"
    f.write_text("import pandas as pd\nimport numpy as np\nfrom sklearn.linear_model import Ridge\nx = 1\n")
    r = run(str(f), "--audit")
    assert "VERIFIED PASS" in r.stdout
    assert r.returncode == 0


def test_audit_passes_clean_multifile_project(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("def get_lag(df):\n    return df.shift(1)\n")
    train = tmp_path / "train.py"
    train.write_text("from helper import get_lag\ndf = 1\n")
    r = run(str(tmp_path), "--audit")
    assert "VERIFIED PASS" in r.stdout
    assert r.returncode == 0


def test_audit_unverifiable_on_relative_import(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sub.py").write_text("def feat(): return 1\n")
    main = pkg / "main.py"
    main.write_text("from .sub import feat\nx = feat()\n")
    r = run(str(main), "--audit")
    assert "UNVERIFIABLE - UNSCANNED LOCAL IMPORTS" in r.stdout
    assert r.returncode == 3


def test_audit_unverifiable_on_local_package_import(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("def feat(): return 1\n")
    train = tmp_path / "train.py"
    train.write_text("from pkg.mod import feat\nx = feat()\n")
    r = run(str(train), "--audit")
    assert "UNVERIFIABLE - UNSCANNED LOCAL IMPORTS" in r.stdout
    assert r.returncode == 3


def test_exemption_negative_rolling_center_true():
    code = "z = (df['x'] - df['x'].rolling(5, center=True).mean()) / df['x'].rolling(5, center=True).std()\n"
    diags, _ = analyze_code(code, "neg_center.py")
    codes = {d['code'] for d in diags}
    assert codes & {'T003', 'T009', 'T010'}


def test_exemption_negative_drop_other_column():
    code = "X = df.copy()\nX.drop(columns=['other'], inplace=True)\nmodel.fit(X, df['target'])\n"
    diags, _ = analyze_code(code, "neg_drop.py")
    assert any(d['code'] == 'C002' for d in diags)


def test_exemption_negative_fit_label_as_X():
    code = "label = df['close'].shift(-1)\nmodel.fit(label, label)\n"
    diags, _ = analyze_code(code, "neg_fit.py")
    assert any(d['code'] == 'F001' for d in diags)


def test_exemption_negative_groupby_symbol():
    code = "df['norm'] = df.groupby('symbol')['x'].transform(lambda s: (s - s.mean()) / s.std())\n"
    diags, _ = analyze_code(code, "neg_groupby.py")
    assert any(d['code'] == 'T010' for d in diags)


def test_audit_unverifiable_on_syntax_error(tmp_path):
    f = tmp_path / "syntax_err.py"
    f.write_text("def broken_code(\n")
    r = run(str(f), "--audit")
    assert "AUDIT FAILED" not in r.stdout
    assert "UNVERIFIABLE - SYNTAX ERROR" in r.stdout
    assert r.returncode == 3


def test_c002_drop_unrelated_column_reassigned_flagged():
    code1 = "X = df.drop(columns=['id'])\nmodel.fit(X, df['target'])\n"
    diags1, _ = analyze_code(code1, "test.py")
    assert any(d['code'] == 'C002' for d in diags1), "X = df.drop(columns=['id']) must trigger C002"

    code2 = "X = df.drop(['id'], axis=1)\nmodel.fit(X, df['target'])\n"
    diags2, _ = analyze_code(code2, "test.py")
    assert any(d['code'] == 'C002' for d in diags2), "X = df.drop(['id'], axis=1) must trigger C002"

    code3 = "X = df.copy()\nX = X.drop(columns=['other'])\nmodel.fit(X, df['target'])\n"
    diags3, _ = analyze_code(code3, "test.py")
    assert any(d['code'] == 'C002' for d in diags3), "X = X.drop(columns=['other']) must trigger C002"


def test_c002_target_dropped_and_readded_with_assign_flagged():
    code = "X = df.drop(columns=['target'])\nX = X.assign(target=df['target'])\nmodel.fit(X, df['target'])\n"
    diags, _ = analyze_code(code, "test.py")
    assert any(d['code'] == 'C002' for d in diags), "Re-adding target with assign must trigger C002"


def test_groupby_month_over_all_years_flagged():
    code = "df['norm'] = df.groupby('month')['x'].transform(lambda s: (s - s.mean()) / s.std())\n"
    diags, _ = analyze_code(code, "test.py")
    assert any(d['code'] == 'T010' for d in diags), "groupby('month') across all years must trigger T010"


def test_audit_unverifiable_on_globals(tmp_path):
    f = tmp_path / "globals_case.py"
    f.write_text("d = globals()\n")
    r = run(str(f), "--audit")
    assert "UNVERIFIABLE - DYNAMIC CODE" in r.stdout
    assert r.returncode == 3

def test_causal_rolling_zscore_is_clean():
    """Causal rolling z-score (x - roll.mean()) / roll.std() does not trigger T003."""
    code = "z = (df['x'] - df['x'].rolling(20).mean()) / df['x'].rolling(20).std()\n"
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'T003'], "Causal rolling z-score should not trigger T003"


def test_bare_rolling_triggers_t003_warning():
    """Bare .rolling().mean() triggers T003 as a warning."""
    code = "df['roll'] = df['price'].rolling(10).mean()\n"
    diags, _ = analyze_code(code, "test.py")
    t003_diags = [d for d in diags if d['code'] == 'T003']
    assert len(t003_diags) == 1
    assert t003_diags[0]['severity'] == 'warning'


def test_c002_reassigned_drop_target_clean():
    """X = df.drop(columns=['target']) followed by fit(X, df['target']) is clean."""
    code = "X = df.drop(columns=['target'])\nmodel.fit(X, df['target'])\n"
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'C002'], "Target was dropped, so C002 must not trigger"


def test_c002_reassigned_drop_positional_axis1_clean():
    """X = df.drop(['target'], axis=1) followed by fit(X, df['target']) is clean."""
    code = "X = df.drop(['target'], axis=1)\nmodel.fit(X, df['target'])\n"
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'C002'], "Target was dropped via axis=1, C002 must not trigger"

def test_c002_unrelated_assign_after_drop_stays_clean():
    """Drop of target followed by unrelated column assignment like X['ret_z'] = ... stays clean."""
    code = """
X = df.drop(columns=['target'])
X['ret_z'] = df['close'] * 2
model.fit(X, df['target'])
"""
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'C002'], "Unrelated column assignment after drop must stay clean"


def test_c002_drop_on_different_frame_flagged():
    """Drop of target on df2 does not remove target from X."""
    code = """
df2 = df.copy()
df2.drop(columns=['target'], inplace=True)
X = df.copy()
model.fit(X, df['target'])
"""
    diags, _ = analyze_code(code, "test.py")
    assert any(d['code'] == 'C002' for d in diags), "Drop on df2 must not count as removing target from X"


def test_t003_differing_rolling_windows_causal_zscore_clean():
    """Differing windows with min_periods and closed='both' is causal and clean."""
    code = "z = (df['x'] - df['x'].rolling(20, min_periods=5).mean()) / df['x'].rolling(60, closed='both').std()\n"
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'T003'], "Differing causal rolling windows must stay clean"


def test_t003_center_true_in_either_window_flagged():
    """center=True in either window of rolling z-score is flagged."""
    code1 = "z = (df['x'] - df['x'].rolling(20, center=True).mean()) / df['x'].rolling(60).std()\n"
    diags1, _ = analyze_code(code1, "test1.py")
    codes1 = {d['code'] for d in diags1}
    assert codes1 & {'T003', 'T009'}, "center=True in first term must be flagged"

    code2 = "z = (df['x'] - df['x'].rolling(20).mean()) / df['x'].rolling(60, center=True).std()\n"
    diags2, _ = analyze_code(code2, "test2.py")
    codes2 = {d['code'] for d in diags2}
    assert codes2 & {'T003', 'T009'}, "center=True in second term must be flagged"


def test_groupby_date_and_month_stays_clean():
    """groupby(['date', 'month']) has a temporal date key and stays clean."""
    code = "df['norm'] = df.groupby(['date', 'month'])['x'].transform(lambda s: (s - s.mean()) / s.std())\n"
    diags, _ = analyze_code(code, "test.py")
    assert not [d for d in diags if d['code'] == 'T010'], "groupby(['date', 'month']) must stay clean"


def test_audit_locals_logging_helper_clean(tmp_path):
    """locals() and vars() inside logging helpers do not invalidate audit certificate."""
    f = tmp_path / "logging_helper.py"
    f.write_text("""
def log_step(name):
    print("Executing", name, locals())
    return vars()
""")
    r = run(str(f), "--audit")
    assert "[VERIFIED PASS]" in r.stdout
    assert r.returncode == 0


def test_audit_locals_subscript_unverifiable(tmp_path):
    """Dynamic lookup via locals()['var'] makes audit unverifiable."""
    f = tmp_path / "dyn_lookup.py"
    f.write_text("x = locals()['feature']\n")
    r = run(str(f), "--audit")
    assert "UNVERIFIABLE - DYNAMIC CODE" in r.stdout
    assert r.returncode == 3

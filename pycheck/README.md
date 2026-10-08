# PyCheck — ML Safety Analyzer

**30 rules. Zero dependencies. Catches the bugs that cost millions.**

PyCheck is a static AST analyzer that detects **temporal lookahead bias**, **causal confusion**, and **unguarded uncertainty** in Python ML and trading scripts — before you run a single line of code.

```bash
pip install pycheck-neuron
pycheck my_strategy.py
```

---

## The Problem It Solves

These bugs are invisible to Python, pytest, and your IDE. They only show up in production — often after you've already lost money or shipped a flawed model.

**Lookahead bias** — the single most expensive silent bug in quantitative ML:
```python
# This looks fine. Python won't complain. Your backtest will show 94% accuracy.
# In production, you lose everything — because this uses data from the future.
df['target'] = df['close'].shift(-1)   # ERROR[T001]: future data access
X_scaled = scaler.fit_transform(X)     # ERROR[T010]: test set leaks into training
cv = KFold(n_splits=5, shuffle=True)   # ERROR[T012]: shuffles time — leaks future
```

PyCheck catches all three before execution:

```
=================================================================
  PyCheck — NEURON ML Safety Analyzer
  Scanning: strategy.py
  Rules: 30 active
=================================================================

  ERROR[T001]: .shift(-1) accesses data 1 rows INTO THE FUTURE
  --> strategy.py:4:14
     4 |  df['target'] = df['close'].shift(-1)
                       ^^^^^^^^^^^
       help: Use .shift(1) to access past data instead

  ERROR[T010]: .fit_transform() on full dataset leaks test statistics into training
  --> strategy.py:8:11
     8 |  X_scaled = scaler.fit_transform(X)
                  ^^^^^^^^^^^
       help: Use .fit(X_train).transform(X_train), then .transform(X_test) separately

  ERROR[T012]: KFold() shuffles time-series data across folds, leaking future into training
  --> strategy.py:11:5
    11 |  cv = KFold(n_splits=5, shuffle=True)
              ^^^^^^^^^^^
       help: Use TimeSeriesSplit() for temporal cross-validation

-----------------------------------------------------------------
  Summary: 3 error(s), 0 warning(s)
  These 3 error(s) would be COMPILE-TIME ERRORS in NEURON
  Python detected: 0 of these issues at runtime
-----------------------------------------------------------------
```

---

## What It Catches

### Temporal Leak Rules (15 rules — `T001`–`T015`)

| Rule | What It Catches | Severity |
|------|----------------|----------|
| `T001` | `.shift(-1)` — accessing future data | **Error** |
| `T002` | `train_test_split()` on time-series data | **Error** |
| `T003` | `.rolling()` computed before the train/test split | Warning |
| `T004` | `.expanding()` computed before split | Warning |
| `T005` | `.pct_change(-1)` — future percent returns | **Error** |
| `T006` | `iloc[i + 1]` inside a loop — future index access | **Error** |
| `T007` | `.bfill()` — backward fill uses future to fill past | **Error** |
| `T008` | `.interpolate(method='cubic')` — non-causal interpolation | Warning |
| `T009` | `.rolling(center=True)` — centered window leaks future | **Error** |
| `T010` | `.fit_transform()` on full dataset — test leaks into train | **Error** |
| `T011` | `StandardScaler` / `MinMaxScaler` fitted before split | **Error** |
| `T012` | `KFold` instead of `TimeSeriesSplit` | **Error** |
| `T013` | `.resample()` on full dataset before split | Warning |
| `T014` | `.diff(-5)` — future difference computation | **Error** |
| `T015` | Negative `.shift()` inside `groupby().transform()` | **Error** |

### Causal Confusion Rules (7 rules — `C001`–`C007`)

| Rule | What It Catches | Severity |
|------|----------------|----------|
| `C001` | `.corr()` used as a basis for trading or treatment decisions | Warning |
| `C002` | Target column included as a feature | **Error** |
| `C003` | Post-treatment variable included in features | Warning |
| `C004` | Filtering by outcome variable — survivorship bias | **Error** |
| `C005` | `.dropna()` removing non-random rows — selection bias | Warning |
| `C006` | p-value threshold without multiple comparison correction | Warning |
| `C007` | `.corrwith()` used for causal feature selection | Warning |

### Uncertainty Rules (6 rules — `U001`–`U006`)

| Rule | What It Catches | Severity |
|------|----------------|----------|
| `U001` | `.predict()` without confidence or probability scores | Warning |
| `U002` | Hardcoded decision threshold (e.g. `if pred > 0.5`) | Warning |
| `U003` | `.predict_proba()` without calibration check | Info |
| `U004` | Single model without ensemble or uncertainty estimate | Info |
| `U005` | `.score()` evaluated on training data only | **Error** |
| `U006` | `.predict()` inside a production loop without guard | **Error** |

### Data Quality Rules (2 rules)

| Rule | What It Catches | Severity |
|------|----------------|----------|
| `D002` | `except: pass` — silently swallowed pipeline errors | Warning |
| `D003` | Magic numbers in data operations | Info |

---

## Taint Flow Analysis

Beyond pattern matching, PyCheck traces how future-contaminated data **flows through your codebase**:

```python
future_ret = df['close'].shift(-1)        # Tainted source
signal     = future_ret - df['close']     # Taint propagates
model.fit(X, signal)                      # ERROR: tainted label reaches .fit() sink
```

The taint engine follows assignments across three levels of variable indirection.

---

## CLI Usage

```bash
pycheck script.py              # Errors + warnings (default)
pycheck script.py --info       # Include info-level diagnostics
pycheck script.py --quiet      # Errors only
pycheck script.py --json       # JSON output — pipe into CI/CD
pycheck --list                 # List all 30 rules
pycheck --help
```

### JSON Output (CI/CD Integration)

```bash
pycheck strategy.py --json | jq '.[] | select(.severity == "error")'
```

Returns structured output for automated pipelines:
```json
[
  {
    "line": 22,
    "col": 14,
    "severity": "error",
    "code": "T001",
    "message": ".shift(-5) accesses data 5 rows INTO THE FUTURE",
    "help": "Use .shift(5) to access past data instead"
  }
]
```

### Exit Codes

- `0` — No errors found
- `1` — One or more errors found

Clean integration with `make`, `pre-commit`, and GitHub Actions.

---

## Real-World Examples

The `examples/` directory contains recreations of actual ML disasters:

| File | What It Demonstrates |
|------|---------------------|
| `zillow_ibuying.py` | 11 bugs replicating the structural failures behind the $881M Zillow Offers write-down |
| `yfinance_strategy.py` | The classic quant blog tutorial: `shift(-5)` + `fit_transform` before split |
| `sentdex_tutorial.py` | A widely-shared ML tutorial with silent time-series cross-validation errors |
| `leaky_pipeline.py` | A complete end-to-end pipeline showing taint propagation across 4 variables |
| `clean_pipeline.py` | The correct version — 0 errors, 0 warnings |

---

## VS Code Extension

The `pycheck-vscode` extension gives you **real-time red/yellow squiggles** as you write:

- Inline error markers on every flagged line
- Hover tooltips with rule description and fix suggestion
- Status bar showing live error/warning count

---

## Beyond Python: the NEURON Language

PyCheck catches these bugs **after you write them**.

[NEURON](https://github.com/neuronlabs-ai/neuron-lang) prevents them **before the program compiles** — by encoding time horizons, causal modes, and uncertainty directly into the type system:

```python
# In NEURON, this is a compile-time error — not a runtime surprise:
fn strategy(prices: Temporal[Tensor, 0]) -> Tensor:
  let future = prices.shift(-1)
  # error[TemporalLeak]: shift(-1) produces offset +1 > 0
  # Lookahead bias detected at compile time.
```

For 100% compile-time temporal proofs, type-checked do-calculus, and native ahead-of-time compilation, see the [NEURON language](https://github.com/neuronlabs-ai/neuron-lang).

---

## License

MIT — free for any use.

---

*PyCheck is built on the [NEURON](https://github.com/neuronlabs-ai/neuron-lang) type theory for temporal, causal, and uncertainty safety in ML programs.*

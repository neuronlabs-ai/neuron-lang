# PyCheck — ML Safety Analyzer

**34 rules. Zero dependencies. Catches the bugs that cost millions.**

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
  Rules: 34 active + Data Flow & Provenance Engine
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

### Temporal Leak Rules (19 rules — `T001`–`T019`)

| Rule | What It Catches | Severity |
|------|----------------|----------|
| `T001` | `.shift(-1)` — accessing future data (including bound-method aliases like `sh = df.shift; sh(-1)`) | **Error** |
| `T002` | `train_test_split()` on time-series data without chronological ordering | **Error** |
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
| `T016` | `pd.merge_asof()` with `direction='nearest'` or `'forward'` | **Error** |
| `T017` | Target laundering & un-split target encoding (`regime_success_rate`) | **Error** |
| `T018` | In-sample target encoding via `.transform()` or un-split group means | **Error** |
| `T019` | `np.roll(arr, -k)` — negative shift rotates future data into present position | **Error** |

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

## Data Flow & Provenance Tracking Engine

Beyond syntactic pattern matching, PyCheck includes a two-pass AST data flow engine that traces data lineage across reassignments, aliases, joins, and helper functions:

### 1. Future Taint Propagation (`F001`)
Traces future-leaked signals across multi-level variable indirection:
```python
future_ret = df['close'].shift(-1)        # Future taint source
signal     = future_ret - df['close']     # Taint propagates through arithmetic
model.fit(X, signal)                      # ERROR[F001]: tainted data reaches .fit() sink
```

### 2. Partition Provenance & Contamination (`F002`)
Identifies training partitions, test partitions, and cross-fold leakage:
```python
# Naming conventions alone are insufficient when variables get reassigned:
X_train = pd.concat([X_train, X_test])    # Tainted with test fold data
scaler.fit_transform(X_train)             # ERROR[F002]: test partition data in training sink
```

PyCheck follows:
- **Concatenation & Merges**: `pd.concat`, `np.concatenate`, `.append()`, `.merge()`, `.join()`.
- **Aliases**: `X_eval = X_test` followed by combination or fitting.
- **Inter-Procedural Helper Functions**: Functions returning expressions derived from test partition arguments.

---

## CLI Usage

```bash
pycheck script.py              # Errors + warnings (default)
pycheck script.py --info       # Include info-level diagnostics
pycheck script.py --quiet      # Errors only
pycheck script.py --json       # JSON output — pipe into CI/CD
pycheck --list                 # List all 34 rules
pycheck --help
```

### Inline Rule Suppression

PyCheck supports standard inline suppression comments with audit reason annotations:

```python
# Specific rule suppression
df['target'] = df['close'].shift(-1)  # pycheck: ignore[T001]

# Multi-rule suppression with audit justification
signal = df[['close', 'target']].corr()  # pycheck: ignore[C001, C002] reason: exploratory EDA baseline

# Blanket ignore or flake8-style noqa
cv = KFold(n_splits=5, shuffle=True)  # pycheck: ignore
X_tr = scaler.fit_transform(X)         # noqa: T010
```

Both same-line trailing comments and line-above comments are supported.

### Sink-Gated Precision Engine

To prevent false alarms during research and exploratory data analysis (EDA), PyCheck uses **AST sink-gating**:
- Exploratory display calls like `df[['feature', 'target']].head()`, `.plot()`, `.describe()`, and `print(...)` are recognized as safe inspection and will **never** trigger feature leakage (`C002`).
- The rule strictly guards actual modeling sinks (`X = ...`, `model.fit(...)`, `cross_validate(...)`).

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

Clean integration with `make`, `pre-commit`, GitHub Actions, and GitLab CI.

---

## Enterprise CI/CD & GitLab Integration

PyCheck natively exports into the **GitLab Code Quality** report format (`gl-codequality.json`), enabling quantitative teams, hedge funds, and enterprise banks to guard against lookahead bias directly in their Merge Requests.

```bash
# Output GitLab Code Quality JSON directly to stdout
pycheck . --gitlab

# Or write report artifact while printing directory summary in CI job trace
pycheck . --gitlab --output gl-codequality.json
```

### Turnkey `.gitlab-ci.yml`

Add this job to your repository to automatically block any Merge Request containing lookahead leaks:

```yaml
pycheck:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install pycheck-neuron
  script:
    - pycheck . --gitlab --output gl-codequality.json
  artifacts:
    reports:
      codequality: gl-codequality.json
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  allow_failure: false
```

GitLab will automatically annotate the Merge Request diff with red warnings on every line where temporal data leaks or unguarded point predictions are introduced.

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

---

## PyCheck Cloud & Enterprise SaaS

For institutional hedge funds, algorithmic trading desks, and enterprise ML engineering teams, **PyCheck Cloud** provides continuous compliance and automated gatekeeping:

* **Automated PR / MR Gatekeeping**: GitHub App & GitLab integration that automatically blocks pull requests containing temporal leaks, lookahead joins, or cross-fold contamination before they reach production.
* **Proof-of-Causality Audit Certificates**: Cryptographically verifiable backtest audit reports (PDF & JSON) certifying that quantitative strategies and feature pipelines are provably free of temporal lookahead bias for LP allocators and risk committees.
* **Firm-Wide Policy Engine**: Author proprietary firm-specific invariants, restrict internal data feed schemas, and monitor leak incident trends across 50+ quant researchers.
* **Air-Gapped & On-Prem Deployment**: Private VPC and on-premise Kubernetes support for proprietary trading strategies that cannot leave your internal network.

> See the [Enterprise CI/CD Integration](#enterprise-cicd--gitlab-integration) section or visit [https://github.com/neuronlabs-ai/pycheck](https://github.com/neuronlabs-ai/pycheck) to learn more or request institutional access.

---

## PyCheck & NEURON: Working Hand-in-Hand

PyCheck and the **[NEURON Programming Language](https://github.com/neuronlabs-ai/neuron-lang)** represent a dual-defense ecosystem against algorithmic failures:

| System | Role | Mechanism | Guarantee |
| :--- | :--- | :--- | :--- |
| **PyCheck** | **The Diagnostic Layer** for existing Python code | 2-pass AST data flow & provenance static analysis | Flags known leakage patterns & contamination in seconds |
| **NEURON** | **The Permanent Cure** for future algorithmic systems | Type-theoretic stochastic filtration proofs ($\mathcal{F}_t$) | **100% mathematical compile-time proof**; zero runtime overhead |

### The Migration Pathway
1. **Audit Today with PyCheck**: Run `pycheck` across existing Python/Pandas/PyTorch repositories to identify lookahead leaks and target laundering.
2. **Transpile with `py2nr`**: Use the automated transpiler to convert validated feature pipelines into typed NEURON modules.
3. **Deploy with 100% Compile-Time Safety**:
```python
# In NEURON, lookahead bias is a compile-time type error — not a runtime surprise:
fn strategy(prices: Temporal[Tensor, 0]) -> Tensor:
  let future = prices.shift(1)  # NEURON uses causal algebraic offsets (+1 is future)
  # error[TemporalLeak]: operation requires filtration F_{t+1}, but context is F_t (offset 0)
  # Lookahead bias rejected at compile time before execution.
```

Explore the language: [https://github.com/neuronlabs-ai/neuron-lang](https://github.com/neuronlabs-ai/neuron-lang)

---

## License

MIT — free for any use.

---

*PyCheck is built on the [NEURON](https://github.com/neuronlabs-ai/neuron-lang) type theory for temporal, causal, and uncertainty safety in ML programs.*

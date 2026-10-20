# PyCheck — VS Code ML Safety Analyzer

<p align="center">
  <img src="https://raw.githubusercontent.com/neuronlabs-ai/pycheck/main/logo.jpg" alt="PyCheck Logo" width="128" height="128" />
</p>

<p align="center">
  <strong>Real-time detection of temporal lookahead bias, causal confusion, and data leakage directly in your editor.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycheck-neuron/"><img src="https://img.shields.io/pypi/v/pycheck-neuron?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://github.com/neuronlabs-ai/pycheck"><img src="https://img.shields.io/badge/rules-33%20active-blueviolet" alt="Rules"></a>
  <a href="https://github.com/neuronlabs-ai/pycheck/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <img src="https://img.shields.io/badge/dependencies-0-success" alt="Zero Dependencies">
</p>

---

## Why PyCheck?

Standard linters (`flake8`, `pylint`, `ruff`) check for syntax, unused imports, and formatting. Python itself runs lookahead code without throwing a single exception. 

As a result, quantitative researchers and ML engineers often spend months tuning strategies that produce **artificially inflated backtests** (e.g. 50%+ CAGR or 95% accuracy), only to lose capital when deployed live.

**PyCheck catches these silent errors inside VS Code in real-time — before you run a single backtest.**

```python
# Standard linters see nothing wrong here.
# PyCheck highlights these with instant red squiggly lines:

df['target'] = df['close'].shift(-1)   # ❌ ERROR[T001]: Future data access
X_scaled = scaler.fit_transform(X)     # ❌ ERROR[T010]: Test data leaks into train
cv = KFold(n_splits=5, shuffle=True)   # ❌ ERROR[T012]: Shuffles time-series data
df['enc'] = df.groupby('cat')['y'].transform('mean') # ❌ ERROR[T018]: In-sample self-target leak
```

---

## Key Features

- **⚡ Instant Real-Time Analysis**: Runs on file open, file save (`Ctrl+S`), or tab switch.
- **🔍 Actionable Hover Diagnostics**: Hover over any red or yellow squiggle to see the mathematical explanation and the drop-in remediation code.
- **📋 Problems Panel Integration**: All findings appear neatly organized under the VS Code Problems panel (`Ctrl+Shift+M`).
- **🛡️ Status Bar Indicator**: Click `$(shield) PyCheck` in the bottom status bar anytime to manually trigger an audit.
- **🔒 100% Offline & Private**: All AST analysis runs locally on your machine. Zero telemetry, zero cloud calls, zero external dependencies.

---

## Getting Started

### 1. Install the Python Engine
PyCheck for VS Code uses the lightweight, zero-dependency `pycheck-neuron` Python package:

```bash
pip install -U pycheck-neuron
```

### 2. Open any Python ML or Trading Script
That's it! Open any `.py` file containing pandas, scikit-learn, numpy, or trading code. PyCheck immediately begins scanning for leakage.

---

## What It Catches (33 Rules)

### Temporal Leak Rules (18 Rules)
* **`T001`**: `.shift(-1)` — negative shift accesses future rows.
* **`T002`**: Random `train_test_split()` on time-series data.
* **`T003`**: `.rolling()` computed before partition splitting.
* **`T004`**: `.expanding()` computed before partition splitting.
* **`T005`**: `.pct_change(-1)` — future return calculations.
* **`T006`**: `iloc[i + 1]` future index lookups inside iterative loops.
* **`T007`**: `.bfill()` — backward fill leaking future values into past NaNs.
* **`T008`**: Non-causal interpolation (`method='cubic'`).
* **`T009`**: `.rolling(center=True)` — centered windows leaking future context.
* **`T010`**: `.fit_transform()` on un-split dataset.
* **`T011`**: Scalers fitted on test or full dataset.
* **`T012`**: `KFold` instead of temporal splits (`TimeSeriesSplit`).
* **`T013`**: `.resample()` on full dataset before splitting.
* **`T014`**: Future `.diff(-n)` access.
* **`T015`**: Negative shift inside `groupby().transform()`.
* **`T016`**: `merge_asof` with forward or nearest direction.
* **`T017`**: Target laundering across un-split partitions.
* **`T018`**: **In-sample self-target encoding leaks** (`groupby().transform('mean')`, `TargetEncoder(cv=None)`).

### Causal Confusion Rules (7 Rules)
* **`C001`**: `.corr()` used as decision criteria for trading or treatment.
* **`C002`**: Target column included in feature matrix inputs.
* **`C003`**: Conditioning on post-treatment colliders/mediators.
* **`C004`**: Selection on outcome variable (survivorship bias).
* **`C005`**: Non-random `.dropna()` causing sample selection bias.
* **`C006`**: Multiple hypothesis testing without Family-Wise Error Rate correction.
* **`C007`**: `.corrwith()` used for causal feature selection.

### Uncertainty & Data Quality Rules (8 Rules)
* **`U001`**: `.predict()` without confidence / probability calibration.
* **`U002`**: Hardcoded decision boundaries (e.g. `if pred > 0.5`).
* **`U003`**: `.predict_proba()` without Brier score or calibration curve.
* **`U004`**: Single point estimate model without ensemble or uncertainty bounds.
* **`U005`**: Model evaluation evaluated on training partition only.
* **`U006`**: Unguarded `.predict()` inside real-time loop.
* **`D002`**: Silent exceptions (`except: pass`) hiding pipeline corruption.
* **`D003`**: Undocumented magic constants in financial calculations.

---

## Configuration Settings

You can customize PyCheck in your VS Code `settings.json`:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `pycheck.enabled` | `true` | Enable or disable PyCheck analysis |
| `pycheck.pythonPath` | `"python"` | Custom path to your Python interpreter (e.g. inside a conda or venv) |
| `pycheck.showInfo` | `false` | Surface informational architectural tips alongside errors and warnings |

---

## License & Community

- **License**: MIT
- **GitHub Repository**: [github.com/neuronlabs-ai/pycheck](https://github.com/neuronlabs-ai/pycheck)
- **PyPI Package**: [pypi.org/project/pycheck-neuron](https://pypi.org/project/pycheck-neuron)
- **Language Origin**: Built as the Python companion to [NEURON Zero-Leakage Programming Language](https://github.com/neuronlabs-ai/neuron-lang).

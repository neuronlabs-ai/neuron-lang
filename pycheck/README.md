# PyCheck — NEURON ML Safety Analyzer

**30 rules. Zero dependencies. Catches bugs Python can't.**

PyCheck is a static AST analyzer that catches **temporal lookahead bias**, **causal confusion**, and **unguarded uncertainty** in Python ML and trading scripts — before you run a single line of code.

## Quickstart

```bash
pip install pycheck-neuron
pycheck examples/comprehensive_leakage.py
```


## What It Catches

### Temporal Leak Detection (15 rules)
| Rule | What It Catches | Severity |
|------|----------------|----------|
| T001 | `.shift(-1)` — future data access | Error |
| T002 | `train_test_split()` on time series | Error |
| T003 | `.rolling()` before train/test split | Warning |
| T004 | `.expanding()` before split | Warning |
| T005 | `.pct_change(-1)` — future returns | Error |
| T006 | `iloc[i+1]` in loop — future index | Error |
| T007 | `.bfill()` — backward fill leaks future | Error |
| T008 | `.interpolate(method='cubic')` — non-causal | Warning |
| T009 | `.rolling(center=True)` — centered window | Error |
| T010 | `.fit_transform()` on full dataset | Error |
| T011 | `StandardScaler` before split | Error |
| T012 | `KFold` instead of `TimeSeriesSplit` | Error |
| T013 | `.resample()` before split | Warning |
| T014 | `.diff(-1)` — future difference | Error |
| T015 | Negative shift inside `groupby().transform()` | Error |

### Causal Confusion Detection (7 rules)
| Rule | What It Catches | Severity |
|------|----------------|----------|
| C001 | `.corr()` used for causal decisions | Warning |
| C002 | Target column used as a feature | Error |
| C003 | Post-treatment variable in features | Warning |
| C004 | Filtering by outcome (survivorship bias) | Error |
| C005 | `.dropna()` non-random data loss | Warning |
| C006 | p-value threshold without correction | Warning |
| C007 | `.corrwith()` for feature selection | Warning |

### Uncertainty Detection (6 rules)
| Rule | What It Catches | Severity |
|------|----------------|----------|
| U001 | `.predict()` without confidence scores | Warning |
| U002 | Hardcoded prediction threshold | Warning |
| U003 | `.predict_proba()` without calibration check | Info |
| U004 | Single model without ensemble | Info |
| U005 | `.score()` on training data only | Error |
| U006 | `.predict()` in production loop | Warning |

### Data Quality (2 rules)
| Rule | What It Catches | Severity |
|------|----------------|----------|
| D002 | `except: pass` — silent error swallowing | Warning |
| D003 | Magic numbers in data operations | Info |

### Data Flow Analysis
PyCheck includes a **taint propagation engine** that tracks how future-leaked data flows through assignments:
```python
future = df['close'].shift(-1)    # Tainted source
signal = future - df['close']      # Taint propagates
model.fit(X, signal)               # ERROR: tainted data reaches training sink
```

## CLI Options

```bash
pycheck script.py              # Default: errors + warnings
pycheck script.py --info       # Include info-level diagnostics
pycheck script.py --quiet      # Errors only
pycheck script.py --json       # JSON output for CI/CD
pycheck --list                 # List all rules
pycheck --help                 # Help
```

## VSCode Extension

Install the `pycheck-vscode` extension for real-time diagnostics:
- Red/yellow squiggles on temporal leaks as you type
- Hover tooltips explaining each bug
- Status bar showing error/warning count

## Example Output

```
=================================================================
  PyCheck — NEURON ML Safety Analyzer
  Scanning: strategy.py
  Rules: 30 active
=================================================================

  ERROR[T001]: .shift(-1) accesses data 1 rows INTO THE FUTURE
  --> strategy.py:9:25
     9 |  df['future'] = df['close'].shift(-1)
                        ^^^^^^^^^^^
       help: Use .shift(1) to access past data instead

  ERROR[T006]: Indexing [i + 1] inside loop accesses future data
  --> strategy.py:21:21
    21 |      next_price = df['close'].iloc[i + 1]
                         ^^^^^^^^^^^
       help: Only use [i] or [i - n] to access current/past data

-----------------------------------------------------------------
  Summary: 2 error(s), 1 warning(s), 0 info(s)
  These 2 error(s) would be COMPILE-TIME ERRORS in NEURON
  Python detected: 0 of these issues at runtime
-----------------------------------------------------------------
```

## Beyond Python: NEURON Language

PyCheck catches these bugs **after you write them**. NEURON prevents them **before they compile**:

```neuron
// This is a COMPILE-TIME ERROR in NEURON:
fn strategy(prices: Temporal[Tensor, past_to_future]):
  let future = prices.shift(-1)  // ERROR: temporal direction violation
```

For 100% compile-time temporal proofs, type-checked causal do-calculus, and native AOT execution, see [NEURON](https://github.com/neuronlabs-ai/neuron-lang).

## License

MIT License


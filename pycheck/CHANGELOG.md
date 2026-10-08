# Changelog

All notable changes to `pycheck-neuron` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-10-05

### Added
- **30 rules** across 4 categories: 15 Temporal (`T001`–`T015`), 7 Causal (`C001`–`C007`), 6 Uncertainty (`U001`–`U006`), 2 Data Quality (`D002`, `D003`)
- **2-pass taint flow engine** (`flow.py`): traces future-leaked data through variable assignments into `.fit()` / `.predict()` sinks
- **`--json` output** for CI/CD pipeline integration
- **`--list`** subcommand for rule enumeration
- **`--quiet`** flag for errors-only output
- **Exit code 1** when any errors are found (clean `make`/CI integration)
- `leaky_pipeline.py` — canonical 8-bug demo example with inline annotations
- `clean_pipeline.py` — the corrected counterpart (0 errors, 0 warnings)
- `zillow_ibuying.py` — recreation of structural ML failures behind the $881M Zillow Offers write-down (11 detected errors)
- `yfinance_strategy.py` / `sentdex_tutorial.py` — real-world quant blog patterns
- VS Code extension (`pycheck-vscode`) for real-time inline diagnostics

### Rule Details
- `T001` — detects `.shift(-n)`, `shift(periods=-n)`, `getattr(obj, 'shift')(-n)` patterns
- `T006` — detects `iloc[i + 1]` and `[i + 1]` subscripts inside `for`/`while` loops
- `T010` — detects `.fit_transform()` and `.fit()` before a train/test split
- `T015` — detects `.shift(-n)` inside `groupby().transform()` chains
- `C002` — detects when the target/label column appears in the feature matrix
- `C004` — detects filtering on an outcome column (survivorship bias)
- `U005` — detects `.score(X_train, y_train)` (training set evaluation presented as accuracy)

### Quality
- 79 unit tests, 100% pass rate
- Zero runtime dependencies (pure Python stdlib + `ast`)
- Supports Python 3.7–3.12

---

## [0.2.0] — 2026-09-05

### Added
- Initial 30-rule implementation
- Basic taint tracking
- CLI with `--json`, `--quiet`, `--info` flags

---

## [0.1.0] — 2026-09-04

- Prototype release with T001–T005 and C001–C003

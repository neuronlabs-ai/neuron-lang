# Changelog

All notable changes to `pycheck-neuron` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.7] — 2026-10-24

### Fixed
- **Target In Features Drop Verification (C002)**:
  - Enhanced dataframe lineage tracking across variable assignments, copies, and intermediate transformations.
  - Fixed drop leniency: reassigned drops of unrelated columns (e.g. `X = df.drop(columns=['id'])` or `X = df.drop(['id'], axis=1)`) followed by `fit(X, df['target'])` are strictly flagged because the target remains in `X`.
  - Added support for target removal via `del X['target']`, `del X[target_col]`, `X.pop('target')`, `drop(columns=[target_col])`, and explicit column selection (`X = df[['feat1', 'feat2']]`).
  - Added evasion prevention: re-adding the target column downstream via `X.assign(target=...)` or `X['target'] = ...` after a prior drop is detected and flagged.
- **Audit Certificate Syntax Error Handling**:
  - Syntax errors in audited files are now cleanly classified as `[UNVERIFIABLE - SYNTAX ERROR: invalid Python syntax]` with institutional exit code 3, rather than incorrectly declaring audit failure or leakage detection.
- **Causal Rolling Window Normalization (T003)**:
  - Maintained `T003` as a `warning` for bare un-split rolling windows (`df.rolling().mean()`), while exempting causal rolling z-scores / standardizations (`(x - roll.mean()) / roll.std()`).
  - Preserved `T009` strict error flagging for centered lookahead windows (`center=True`).
- **Dynamic Code & Calendar Groupby Auditing**:
  - Added `globals()`, `locals()`, and `vars()` to opaque dynamic construct detection for audit verification gating (exit code 3).
  - Excluded cyclical calendar keywords (`'month'`, `'day'`, `'year'`) from temporal cross-section markers in `T010`, ensuring multi-year cyclical groupings (`df.groupby('month')`) are properly flagged.

## [1.0.6] — 2026-10-24

### Added
- **Target `y` Exemption in Forecasting Sinks (F001)**: In supervised model calls (`fit`, `fit_predict`, `train`, `score`), the target label at position 1 or keyword (`y`, `target`, `label`) is recognized as a valid future forecasting target (e.g. `y = df['close'].shift(-1)`), while remaining strictly forbidden in feature matrices `X` (`fit(label, label)` or `fit(label, y)` are strictly flagged).
- **Causal Windowed & Cross-Sectional Normalization Exemption (T010)**: Normalizations computed over causal windows or cross-sections (`rolling().mean()/.std()`, `expanding()`, `resample()`, `df.groupby('date')['ret'].rank(pct=True)`), including inside `.transform()` and `.apply()` lambdas, are recognized as causal and exempt from full-sample `T010`. Non-causal windows (`center=True` or future-shifted rollings) remain strictly flagged.
- **AST-Based Temporal Context Detection (T012, T002)**: Replaced substring text matching with AST syntactic inspection (checking datetime method calls, date column indexing, and `parse_dates` keywords), eliminating false positive time-series classification on iid scripts containing words like `"flower"`, `"allow"`, or `.lower()`.
- **Full-Sample Scaling & Normalization Leak Detection (T010)**: Added AST pattern matching for un-split full-sample min-max scaling (`(x - min)/(max - min)`), max-scaling (`x / x.max()`), quantile clipping (`.quantile()`), centering (`x - x.mean()`), and normalization inside `.apply(lambda)`.
- **Derived Feature Taint Detection (F001)**: Added tracking for future-tainted variables assigned to feature variables (e.g. `feats = label.rolling(5).mean()`), catching lookahead feature construction even without a local `.fit()` sink.
- **Target Column Alias Tracking (C002)**: Detects target columns riding into feature data via `df.copy()` without dropping the target.
- **Audit Opaque Code Detection**: `--audit` uses AST inspection to catch runtime dynamic code execution (`eval()`, `exec()`), issuing an `[UNVERIFIABLE - DYNAMIC CODE DETECTED]` certificate and non-zero exit code instead of false clean passes.
- **Exemption Regression Suite**: Added comprehensive regression tests verifying that window exemptions never suppress forward-looking windows, centered rollings, negative shifts in normalizations, or tainted feature inputs.

## [1.0.5] — 2026-10-24

### Fixed
- **Headline Demo & Target Column Precision (T001)**: Restricted target suppression strictly to standalone label variable assignments (`isinstance(tgt, ast.Name)`), ensuring dataframe target column shifts (`df['target'] = df['close'].shift(-1)`, `df['y'] = ...`) are never suppressed. Restores exact match with README headline demo.
- **Taint Flow on Columns & Containers (F001)**: Enhanced `TaintTracker` with container propagation (`pd.concat`, `np.concatenate`, `vstack`, `hstack`, `merge`, `append`, `join`) and column subscript tracking (`df['feat'] = label; model.fit(df[['feat']], y)`), ensuring full-feature taint detection across all data structures.
- **Strict Temporal Context over Name Heuristics (T002)**: Removed blind variable name bypass (`iris`, `images`) in favor of strict temporal context analysis. Temporal files using non-standard variable names are now reliably audited for time-series leaks.
- **Full-Sample Preprocessing Statistics Before Split (T010)**:
  - Detects manual full-sample z-score standardization `(df['x'] - df['x'].mean()) / df['x'].std()` prior to train/test split.
  - Detects `.fillna()` with full-sample mean, median, or standard deviation (`df.fillna(df.mean())`).
  - Detects full-sample percentile ranking (`df['x'].rank(pct=True)`).
- **Adversarial Lookahead Patterns**:
  - `T001`: Added reverse-slice cumulative sum lookahead (`df['ret'][::-1].cumsum()[::-1]`) and manual 1-step lookahead leads via numpy slicing (`np.append(arr[1:], np.nan)`).
  - `T007`: Added reverse-slice forward-fill detection (`df.iloc[::-1]['price'].ffill()`).
- **Proof-of-Causality Audit Integrity**:
  - Synchronized audit checklist to all 34 rules (`T001`–`T019`).
  - Added granular breakdown for warnings vs critical errors to ensure accurate verification status.
  - Exposed programmatic `analyze_code(source: str)` API.
- **Test Suite**: Expanded test suite to 142 unit tests (100% pass rate). 23/25 adversarial stress leaks caught (92%), and 13/13 clean isolated cases pass with 0 false positives.

---

## [1.0.2] — 2026-10-17

### Changed
- **Documentation & PyPI Sync**: Updated package documentation, distribution metadata, and PyPI project specifications to explicitly reflect all **32 rules** across all 4 categories (17 Temporal `T001`–`T017`, 7 Causal `C001`–`C007`, 6 Uncertainty `U001`–`U006`, 2 Data Quality `D002`–`D003`).
- **Test Suite Verification**: Updated integration test assertion (`test_rules.py`) to strictly enforce the exact 32-rule invariant.
- **Proof-of-Causality & Server Enhancements**: Integrated cryptographic audit certificate generation (`pycheck --audit`) and automated webhook gatekeeper endpoints.

---

## [1.0.1] — 2026-10-17

### Added
- **Rule Suite Expansion (32 rules total)**: Added rules `T016` (`merge_asof_direction`) and `T017` (`target_laundering`), expanding the active safety suite from 30 to 32 rules.
- **AST Data Provenance Tracking Engine (`flow.py`)**:
  - Full partition provenance tracking across train/test splits, slices, and cross-validation loops.
  - Detects partition contamination when training variables are concatenated or reassigned with test data (`X_train = pd.concat([X_train, X_test])`).
  - Follows aliases (`X_eval = X_test`), `.merge()`, `.append()`, and inter-procedural helper function return lineages.
  - New diagnostic code **`F002`**: Flags data contamination and test partition variables passed to `.fit()`, `.fit_transform()`, and `.train()`.
- **Rule `T016` (`merge_asof_direction`)**: Detects lookahead bias in `pd.merge_asof()` joins using `direction='nearest'` or `'forward'`.
- **Rule `T017` (`target_laundering`)**:
  - Flags target laundering via group aggregations across the un-split dataset (`df.groupby('regime')['target'].transform('mean')`).
  - Validates legitimate fold-local target encoding computed strictly within training folds (`train_df.groupby('regime')['target'].mean()`).
  - Emits diagnostic warnings for test set mappings missing `.fillna(train_prior_mean)` to guard against inference failures on unseen categories/regimes.
- **Rule `C002` Refinement**: Distinguishes fold-local engineered features (`train_df[['volatility', 'regime_success_rate']]`) from raw target columns (`target`, `label`, `future_price`).
- **VS Code Extension v1.0.1**: Upgraded `pycheck-neuron` VS Code extension to v1.0.1 with instant linting for provenance contamination and target laundering.
- **Unit Test Suite Expansion**: Added 16 new unit tests, bringing total test suite to 97 tests with 100% pass rate.

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
- Initial 30-rule prototype implementation (expanded to 32 rules in v1.0.1+)
- Basic taint tracking
- CLI with `--json`, `--quiet`, `--info` flags

---

## [0.1.0] — 2026-09-04

- Prototype release with T001–T005 and C001–C003

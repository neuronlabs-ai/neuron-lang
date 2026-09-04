"""
PyCheck Test Suite — one positive and one negative test per rule.
Run with: pytest tests/test_rules.py -v
"""

import ast
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pycheck.rules import (
    ALL_RULES, AnalysisContext,
    T001_ShiftNegative, T002_TrainTestSplit, T003_RollingBeforeSplit,
    T004_ExpandingBeforeSplit, T005_PctChangeNegative, T006_FutureIndexInLoop,
    T007_BackfillLeak, T008_NonCausalInterpolate, T009_CenteredRolling,
    T010_FitBeforeSplit, T011_ScalerBeforeSplit, T012_KFoldTimeSeries,
    T013_ResampleBeforeSplit, T014_DiffNegative, T015_GroupbyShiftLeak,
    C001_CorrelationCausation, C002_TargetInFeatures, C003_PostTreatmentVariable,
    C004_SelectionOnOutcome, C005_SurvivorshipBias, C006_PHacking,
    C007_CorrWithoutForDecisions,
    U001_PredictNoUncertainty, U002_HardcodedThreshold, U003_NoCalibration,
    U004_NoEnsemble, U005_ScoreOnTrainOnly, U006_PredictInLoop,
    D002_SilentExceptionSwallow, D003_MagicNumbers,
)
from pycheck.flow import TaintTracker
from pycheck.analyzer import analyze_file


def _check(rule, code, expect_hit=True, in_loop=False, loop_var=None):
    """Helper: parse code, walk AST, check if rule fires."""
    tree = ast.parse(code)
    ctx = AnalysisContext(code.split('\n'), "<test>")
    ctx.in_loop = in_loop
    ctx.loop_var = loop_var
    # Populate constants and assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = None
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                        val = node.value.value
                    elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                        if isinstance(node.value.operand, ast.Constant) and isinstance(node.value.operand.value, (int, float)):
                            val = -node.value.operand.value
                    if val is not None:
                        ctx.constants[target.id] = val
                    ctx.assignments[target.id] = node.value
    hits = []
    for node in ast.walk(tree):
        hits.extend(rule.check_node(node, ctx))
    if expect_hit:
        assert len(hits) > 0, f"Rule {rule.code} should fire on:\n{code}"
    else:
        assert len(hits) == 0, f"Rule {rule.code} should NOT fire on:\n{code}\nBut got: {[h.message for h in hits]}"
    return hits


# ═══════════════════════════════════════════════════════
#  TEMPORAL LEAK RULES
# ═══════════════════════════════════════════════════════

class TestT001:
    def test_fires_on_negative_shift(self):
        _check(T001_ShiftNegative(), "df['x'] = df['close'].shift(-1)", expect_hit=True)

    def test_fires_on_negative_int_literal(self):
        _check(T001_ShiftNegative(), "df['x'] = df['close'].shift(-5)", expect_hit=True)

    def test_silent_on_positive_shift(self):
        _check(T001_ShiftNegative(), "df['x'] = df['close'].shift(1)", expect_hit=False)

    def test_fires_on_negative_shift_alias(self):
        code = "future_shift = -1\ndf['x'] = df['close'].shift(future_shift)"
        _check(T001_ShiftNegative(), code, expect_hit=True)

    def test_fires_on_periods_keyword(self):
        _check(T001_ShiftNegative(), "df['x'] = df['close'].shift(periods=-1)", expect_hit=True)

    def test_silent_on_no_shift(self):
        _check(T001_ShiftNegative(), "x = df['close'].mean()", expect_hit=False)


class TestT002:
    def test_fires_on_train_test_split(self):
        _check(T002_TrainTestSplit(), "X_train, X_test = train_test_split(X, y)", expect_hit=True)

    def test_silent_on_manual_split(self):
        _check(T002_TrainTestSplit(), "train = data[:100]; test = data[100:]", expect_hit=False)


class TestT003:
    def test_fires_on_rolling(self):
        _check(T003_RollingBeforeSplit(), "df['sma'] = df['close'].rolling(14).mean()", expect_hit=True)

    def test_silent_on_mean(self):
        _check(T003_RollingBeforeSplit(), "x = df['close'].mean()", expect_hit=False)


class TestT004:
    def test_fires_on_expanding(self):
        _check(T004_ExpandingBeforeSplit(), "df['cum'] = df['close'].expanding().max()", expect_hit=True)

    def test_silent_on_cumsum(self):
        _check(T004_ExpandingBeforeSplit(), "x = df['close'].cumsum()", expect_hit=False)


class TestT005:
    def test_fires_on_negative_pct_change(self):
        _check(T005_PctChangeNegative(), "df['r'] = df['close'].pct_change(-1)", expect_hit=True)

    def test_silent_on_positive_pct_change(self):
        _check(T005_PctChangeNegative(), "df['r'] = df['close'].pct_change(1)", expect_hit=False)


class TestT006:
    def test_fires_on_future_index(self):
        _check(T006_FutureIndexInLoop(), "x = df.iloc[i + 1]",
               expect_hit=True, in_loop=True, loop_var="i")

    def test_silent_on_current_index(self):
        _check(T006_FutureIndexInLoop(), "x = df.iloc[i]",
               expect_hit=False, in_loop=True, loop_var="i")

    def test_silent_outside_loop(self):
        _check(T006_FutureIndexInLoop(), "x = df.iloc[i + 1]",
               expect_hit=False, in_loop=False, loop_var=None)


class TestT007:
    def test_fires_on_bfill(self):
        _check(T007_BackfillLeak(), "df['x'] = df['close'].bfill()", expect_hit=True)

    def test_fires_on_fillna_bfill(self):
        _check(T007_BackfillLeak(), "df['x'] = df['close'].fillna(method='bfill')", expect_hit=True)

    def test_silent_on_ffill(self):
        _check(T007_BackfillLeak(), "df['x'] = df['close'].ffill()", expect_hit=False)

    def test_silent_on_fillna_zero(self):
        _check(T007_BackfillLeak(), "df['x'] = df['close'].fillna(0)", expect_hit=False)


class TestT008:
    def test_fires_on_cubic_interpolate(self):
        _check(T008_NonCausalInterpolate(),
               "df['x'] = df['close'].interpolate(method='cubic')", expect_hit=True)

    def test_silent_on_pad_interpolate(self):
        _check(T008_NonCausalInterpolate(),
               "df['x'] = df['close'].interpolate(method='pad')", expect_hit=False)


class TestT009:
    def test_fires_on_centered_rolling(self):
        _check(T009_CenteredRolling(),
               "df['x'] = df['close'].rolling(10, center=True).mean()", expect_hit=True)

    def test_silent_on_default_rolling(self):
        _check(T009_CenteredRolling(),
               "df['x'] = df['close'].rolling(10).mean()", expect_hit=False)


class TestT010:
    def test_fires_on_fit_transform(self):
        _check(T010_FitBeforeSplit(), "X = scaler.fit_transform(df)", expect_hit=True)

    def test_fires_on_fit(self):
        _check(T010_FitBeforeSplit(), "scaler.fit(X_train)", expect_hit=True)

    def test_silent_on_transform(self):
        _check(T010_FitBeforeSplit(), "X = scaler.transform(X_test)", expect_hit=False)


class TestT011:
    def test_fires_on_standard_scaler(self):
        _check(T011_ScalerBeforeSplit(), "s = StandardScaler()", expect_hit=True)

    def test_fires_on_minmax_scaler(self):
        _check(T011_ScalerBeforeSplit(), "s = MinMaxScaler()", expect_hit=True)

    def test_silent_on_non_scaler(self):
        _check(T011_ScalerBeforeSplit(), "s = RandomForest()", expect_hit=False)


class TestT012:
    def test_fires_on_kfold(self):
        _check(T012_KFoldTimeSeries(), "cv = KFold(n_splits=5)", expect_hit=True)

    def test_fires_on_stratified_kfold(self):
        _check(T012_KFoldTimeSeries(), "cv = StratifiedKFold(n_splits=5)", expect_hit=True)

    def test_silent_on_timeseries_split(self):
        _check(T012_KFoldTimeSeries(), "cv = TimeSeriesSplit(n_splits=5)", expect_hit=False)


class TestT013:
    def test_fires_on_resample(self):
        _check(T013_ResampleBeforeSplit(), "df_m = df.resample('M').mean()", expect_hit=True)

    def test_silent_on_groupby(self):
        _check(T013_ResampleBeforeSplit(), "df_m = df.groupby('month').mean()", expect_hit=False)


class TestT014:
    def test_fires_on_negative_diff(self):
        _check(T014_DiffNegative(), "df['d'] = df['close'].diff(-1)", expect_hit=True)

    def test_silent_on_positive_diff(self):
        _check(T014_DiffNegative(), "df['d'] = df['close'].diff(1)", expect_hit=False)

    def test_fires_on_periods_keyword(self):
        _check(T014_DiffNegative(), "df['d'] = df['close'].diff(periods=-1)", expect_hit=True)

    def test_fires_on_diff_alias(self):
        code = "k = -2\ndf['d'] = df['close'].diff(periods=k)"
        _check(T014_DiffNegative(), code, expect_hit=True)


class TestT015:
    # T015 requires a specific chain pattern — test separately
    def test_silent_on_simple_shift(self):
        _check(T015_GroupbyShiftLeak(), "df['x'] = df['close'].shift(1)", expect_hit=False)


# ═══════════════════════════════════════════════════════
#  CAUSAL CONFUSION RULES
# ═══════════════════════════════════════════════════════

class TestC001:
    def test_fires_on_corr(self):
        _check(C001_CorrelationCausation(), "c = df['a'].corr()", expect_hit=True)

    def test_silent_on_cov(self):
        _check(C001_CorrelationCausation(), "c = df['a'].cov()", expect_hit=False)


class TestC002:
    def test_fires_on_target_in_features(self):
        _check(C002_TargetInFeatures(), "X = df[['sma', 'target', 'vol']]", expect_hit=True)

    def test_fires_on_label_in_features(self):
        _check(C002_TargetInFeatures(), "X = df[['sma', 'label']]", expect_hit=True)

    def test_silent_on_clean_features(self):
        _check(C002_TargetInFeatures(), "X = df[['sma', 'volume', 'rsi']]", expect_hit=False)


class TestC003:
    def test_fires_on_post_treatment(self):
        _check(C003_PostTreatmentVariable(), "df['after_surgery'] = df['bp'] * 2", expect_hit=True)

    def test_silent_on_normal_column(self):
        _check(C003_PostTreatmentVariable(), "df['volume'] = df['vol'] * 100", expect_hit=False)


class TestC004:
    def test_fires_on_profit_filter(self):
        _check(C004_SelectionOnOutcome(), "df = df[df['profit'] > 0]", expect_hit=True)

    def test_silent_on_volume_filter(self):
        _check(C004_SelectionOnOutcome(), "df = df[df['volume'] > 1000]", expect_hit=False)


class TestC005:
    def test_fires_on_dropna(self):
        _check(C005_SurvivorshipBias(), "df = df.dropna()", expect_hit=True)

    def test_silent_on_fillna(self):
        _check(C005_SurvivorshipBias(), "df = df.fillna(0)", expect_hit=False)


class TestC006:
    def test_fires_on_pvalue_check(self):
        _check(C006_PHacking(), "if p_value < 0.05: pass", expect_hit=True)

    def test_silent_on_normal_comparison(self):
        _check(C006_PHacking(), "if count < 10: pass", expect_hit=False)


class TestC007:
    def test_fires_on_corrwith(self):
        _check(C007_CorrWithoutForDecisions(), "x = df.corrwith(df['y'])", expect_hit=True)

    def test_silent_on_corr(self):
        # C007 is specifically for corrwith, not corr (that's C001)
        _check(C007_CorrWithoutForDecisions(), "x = df['a'].corr()", expect_hit=False)


# ═══════════════════════════════════════════════════════
#  UNCERTAINTY RULES
# ═══════════════════════════════════════════════════════

class TestU001:
    def test_fires_on_predict(self):
        _check(U001_PredictNoUncertainty(), "y = model.predict(X)", expect_hit=True)

    def test_silent_on_predict_proba(self):
        _check(U001_PredictNoUncertainty(), "y = model.predict_proba(X)", expect_hit=False)


class TestU002:
    def test_fires_on_prediction_threshold(self):
        _check(U002_HardcodedThreshold(), "if prediction > 0.5: buy()", expect_hit=True)

    def test_silent_on_count_threshold(self):
        _check(U002_HardcodedThreshold(), "if count > 10: process()", expect_hit=False)


class TestU003:
    def test_fires_on_predict_proba(self):
        _check(U003_NoCalibration(), "p = model.predict_proba(X)", expect_hit=True)

    def test_silent_on_predict(self):
        _check(U003_NoCalibration(), "p = model.predict(X)", expect_hit=False)


class TestU004:
    def test_fires_on_linear_regression(self):
        _check(U004_NoEnsemble(), "m = LinearRegression()", expect_hit=True)

    def test_silent_on_random_forest(self):
        _check(U004_NoEnsemble(), "m = RandomForestClassifier()", expect_hit=False)


class TestU005:
    def test_fires_on_score_train(self):
        _check(U005_ScoreOnTrainOnly(), "s = model.score(X_train, y_train)", expect_hit=True)

    def test_silent_on_score_test(self):
        _check(U005_ScoreOnTrainOnly(), "s = model.score(X_test, y_test)", expect_hit=False)


class TestU006:
    def test_fires_on_predict_in_loop(self):
        _check(U006_PredictInLoop(), "y = model.predict(X)",
               expect_hit=True, in_loop=True)

    def test_silent_on_predict_outside_loop(self):
        _check(U006_PredictInLoop(), "y = model.predict(X)",
               expect_hit=False, in_loop=False)


# ═══════════════════════════════════════════════════════
#  DATA QUALITY RULES
# ═══════════════════════════════════════════════════════

class TestD002:
    def test_fires_on_bare_except_pass(self):
        code = "try:\n    x = 1\nexcept:\n    pass"
        _check(D002_SilentExceptionSwallow(), code, expect_hit=True)

    def test_silent_on_specific_except(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    pass"
        _check(D002_SilentExceptionSwallow(), code, expect_hit=False)


class TestD003:
    def test_silent_on_small_number(self):
        _check(D003_MagicNumbers(), "df['x'] = df['close'].rolling(14).mean()", expect_hit=False)


# ═══════════════════════════════════════════════════════
#  DATA FLOW / TAINT PROPAGATION
# ═══════════════════════════════════════════════════════

class TestFlow:
    def test_taint_propagates_through_assignment(self):
        code = """
future = df['close'].shift(-1)
signal = future - df['close']
model.fit(X, signal)
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        assert len(results) > 0, "Taint should propagate from shift(-1) through signal to .fit()"
        assert any(d.code == "F001" for d in results)

    def test_no_taint_on_clean_code(self):
        code = """
sma = df['close'].rolling(14).mean()
model.fit(X_train, y_train)
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        flow_errors = [d for d in results if d.code == "F001"]
        assert len(flow_errors) == 0, "No F001 taint errors on clean code"

    def test_taint_through_three_levels(self):
        code = """
a = df['close'].shift(-1)
b = a * 2
c = b + 1
model.fit(X, c)
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        assert any(d.code == "F001" for d in results), "Taint should propagate through 3 assignment levels"


# ═══════════════════════════════════════════════════════
#  INTEGRATION TEST
# ═══════════════════════════════════════════════════════

class TestIntegration:
    def test_rule_count(self):
        assert len(ALL_RULES) >= 30, f"Expected 30+ rules, got {len(ALL_RULES)}"

    def test_all_rules_have_code(self):
        for rule in ALL_RULES:
            assert rule.code, f"Rule {rule.__class__.__name__} missing code"
            assert rule.name, f"Rule {rule.__class__.__name__} missing name"
            assert rule.category, f"Rule {rule.__class__.__name__} missing category"

    def test_no_duplicate_codes(self):
        codes = [r.code for r in ALL_RULES]
        assert len(codes) == len(set(codes)), f"Duplicate rule codes: {[c for c in codes if codes.count(c) > 1]}"

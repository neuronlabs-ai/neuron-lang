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
    T018_InSampleTargetEncoding, T019_NpRollNegative,
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
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
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
    def test_fires_on_timeseries_train_test_split(self):
        code = "df['close'] = [1, 2]\nX_train, X_test = train_test_split(df, y)"
        _check(T002_TrainTestSplit(), code, expect_hit=True)

    def test_silent_on_shuffle_false(self):
        code = "df['close'] = [1, 2]\nX_train, X_test = train_test_split(df, y, shuffle=False)"
        _check(T002_TrainTestSplit(), code, expect_hit=False)

    def test_silent_on_clean_tabular_split(self):
        code = "X_train, X_test = train_test_split(X, y)"
        _check(T002_TrainTestSplit(), code, expect_hit=False)

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
        # scaler.fit(X) — full dataset, leaky — should warn
        _check(T010_FitBeforeSplit(), "scaler.fit(X)", expect_hit=True)

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
    def test_fires_on_kfold_with_temporal_context(self):
        # T012 now requires temporal context markers
        _check(T012_KFoldTimeSeries(), "df['close'] = prices\ncv = KFold(n_splits=5)", expect_hit=True)

    def test_fires_on_stratified_kfold_with_temporal_context(self):
        _check(T012_KFoldTimeSeries(), "df['close'] = prices\ncv = StratifiedKFold(n_splits=5)", expect_hit=True)

    def test_silent_on_timeseries_split(self):
        _check(T012_KFoldTimeSeries(), "cv = TimeSeriesSplit(n_splits=5)", expect_hit=False)

    def test_silent_on_iid_data(self):
        _check(T012_KFoldTimeSeries(), "cv = KFold(n_splits=5, shuffle=True)", expect_hit=False)


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


class TestT018:
    def test_fires_on_groupby_transform_mean(self):
        _check(T018_InSampleTargetEncoding(), "train['enc'] = train.groupby('category')['target'].transform('mean')", expect_hit=True)

    def test_fires_on_groupby_transform_median(self):
        _check(T018_InSampleTargetEncoding(), "df['enc'] = df.groupby('sector')['pnl'].transform('median')", expect_hit=True)

    def test_fires_on_target_encoder_cv_none(self):
        _check(T018_InSampleTargetEncoding(), "enc = TargetEncoder(cv=None)", expect_hit=True)

    def test_fires_on_map_groupby_mean(self):
        _check(T018_InSampleTargetEncoding(), "train['enc'] = train['cat'].map(train.groupby('cat')['target'].mean())", expect_hit=True)

    def test_silent_on_non_target_groupby_transform(self):
        _check(T018_InSampleTargetEncoding(), "train['vol_mean'] = train.groupby('category')['volume'].transform('mean')", expect_hit=False)

    def test_silent_on_target_encoder_with_cv(self):
        _check(T018_InSampleTargetEncoding(), "enc = TargetEncoder(cv=5)", expect_hit=False)

    def test_silent_on_valid_test_mapping(self):
        # Mapping train target statistics onto test set is valid inference
        _check(T018_InSampleTargetEncoding(), "test['enc'] = test['cat'].map(train.groupby('cat')['target'].mean())", expect_hit=False)

    def test_silent_on_valid_oof_val_mapping(self):
        # Mapping fit fold statistics onto val fold in OOF cross-validation is safe
        _check(T018_InSampleTargetEncoding(), "val['enc'] = val['cat'].map(fit.groupby('cat')['target'].mean())", expect_hit=False)


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

    def test_silent_on_exploratory_display(self):
        _check(C002_TargetInFeatures(), "df[['close', 'target']].head()", expect_hit=False)
        _check(C002_TargetInFeatures(), "print(df[['close', 'target']])", expect_hit=False)
        _check(C002_TargetInFeatures(), "df[['close', 'target']].plot()", expect_hit=False)


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
        assert len(ALL_RULES) == 34, f"Expected exactly 34 rules, got {len(ALL_RULES)}"

    def test_all_rules_have_code(self):
        for rule in ALL_RULES:
            assert rule.code, f"Rule {rule.__class__.__name__} missing code"
            assert rule.name, f"Rule {rule.__class__.__name__} missing name"
            assert rule.category, f"Rule {rule.__class__.__name__} missing category"

    def test_no_duplicate_codes(self):
        codes = [r.code for r in ALL_RULES]
        assert len(codes) == len(set(codes)), f"Duplicate rule codes: {[c for c in codes if codes.count(c) > 1]}"

    def test_r2_8_keyword_sink_taint(self):
        code = """
import pandas as pd
df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
x_taint = df.shift(-1)
model.fit(X=x_taint, y=df['y'])
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        assert any(d.code == "F001" and "X" in d.message for d in results), "Keyword arg taint must be detected"

    def test_r2_9_getattr_shift_detected(self):
        code = """
import pandas as pd
df = pd.DataFrame({'x': [1, 2]})
f = getattr(df, 'shift')(-1)
"""
        tree = ast.parse(code)
        rule = T001_ShiftNegative()
    def test_t016_merge_asof_nearest(self):
        code = """
import pandas as pd
merged = pd.merge_asof(left, right, on='ts', direction='nearest')
"""
        tree = ast.parse(code)
        from pycheck.rules import T016_MergeAsofDirection
        rule = T016_MergeAsofDirection()
        diags = []
        for node in ast.walk(tree):
            diags.extend(rule.check_node(node, None))
        assert any(d.code == "T016" for d in diags), "merge_asof with direction='nearest' must trigger T016"

    def test_t016_merge_asof_backward_clean(self):
        code = """
import pandas as pd
merged = pd.merge_asof(left, right, on='ts', direction='backward')
"""
        tree = ast.parse(code)
        from pycheck.rules import T016_MergeAsofDirection
        rule = T016_MergeAsofDirection()
        diags = []
        for node in ast.walk(tree):
            diags.extend(rule.check_node(node, None))
        assert not diags, "merge_asof with direction='backward' must not trigger any diagnostic"

    def test_t010_scaler_on_train_data_clean(self):
        code = """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
scaler.fit(X_train[features])
model.fit(X_train, y_train)
"""
        tree = ast.parse(code)
        from pycheck.rules import T010_FitBeforeSplit
        rule = T010_FitBeforeSplit()
        diags = []
        for node in ast.walk(tree):
            diags.extend(rule.check_node(node, None))
        assert not diags, f"Fitting or fit_transforming on training data must not trigger T010, got: {diags}"

    def test_t010_scaler_on_full_dataset_flagged(self):
        code = """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
"""
        tree = ast.parse(code)
        from pycheck.rules import T010_FitBeforeSplit
        rule = T010_FitBeforeSplit()
        diags = []
        for node in ast.walk(tree):
            diags.extend(rule.check_node(node, None))
        assert any(d.code == "T010" for d in diags), "fit_transform on full dataset must trigger T010"

    def test_t017_regime_success_rate_target_laundering(self):
        code = """
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df['regime_success_rate'] = df.groupby('regime')['target'].transform('mean')
X = df[['volatility', 'regime_success_rate']]
"""
        tree = ast.parse(code)
        from pycheck.rules import T017_TargetLaundering, C002_TargetInFeatures
        t017 = T017_TargetLaundering()
        c002 = C002_TargetInFeatures()
        diags = []
        for node in ast.walk(tree):
            diags.extend(t017.check_node(node, None))
            diags.extend(c002.check_node(node, None))
        assert any(d.code == "T017" for d in diags), "Target groupby aggregation must trigger T017"
        assert any(d.code == "C002" for d in diags), "regime_success_rate feature selection must trigger C002"

    def test_provenance_concat_contamination(self):
        code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.DataFrame({'a': [1, 2]})
X_test = pd.DataFrame({'a': [3, 4]})

X_train = pd.concat([X_train, X_test])
scaler = StandardScaler()
scaler.fit_transform(X_train)
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        assert any(d.code == "F002" for d in results), "Reassigned X_train with concatenated test data must trigger F002"
        f002 = [d for d in results if d.code == "F002"][0]
        assert "X_train" in f002.message
        assert "test partition" in f002.message

    def test_provenance_alias_and_reassignment(self):
        code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler

X_eval = X_test
X_train = pd.concat([X_train, X_eval])
scaler = StandardScaler()
scaler.fit(X_train)
"""
        tree = ast.parse(code)
        tracker = TaintTracker()
        results = tracker.analyze(tree, code.split('\n'))
        assert any(d.code == "F002" for d in results), "Alias of test data in concat must trigger F002"

    def test_provenance_helper_function(self):
        code_leaky = """
import pandas as pd
from sklearn.preprocessing import StandardScaler

def combine_folds(tr, te):
    return pd.concat([tr, te])

X_train = combine_folds(X_train, X_test)
scaler = StandardScaler()
scaler.fit_transform(X_train)
"""
        tree_leaky = ast.parse(code_leaky)
        tracker = TaintTracker()
        results_leaky = tracker.analyze(tree_leaky, code_leaky.split('\n'))
        assert any(d.code == "F002" for d in results_leaky), "Helper function combining test data must trigger F002"

        code_clean = """
from sklearn.preprocessing import StandardScaler

def clean_data(data):
    return data.dropna()

clean_train = clean_data(X_train)
scaler = StandardScaler()
scaler.fit_transform(clean_train)
"""
        tree_clean = ast.parse(code_clean)
        tracker_clean = TaintTracker()
        results_clean = tracker_clean.analyze(tree_clean, code_clean.split('\n'))
        assert not any(d.code == "F002" for d in results_clean), "Clean helper function on training data must not trigger F002"

    def test_t017_fold_local_target_encoding_clean(self):
        code = """
# Legitimate fold-local target encoding
regime_stats = train_df.groupby('regime')['target'].mean()
global_prior = train_df['target'].mean()

train_df['regime_success_rate'] = train_df['regime'].map(regime_stats)
test_df['regime_success_rate'] = test_df['regime'].map(regime_stats).fillna(global_prior)
X_train = train_df[['volatility', 'regime_success_rate']]
"""
        tree = ast.parse(code)
        from pycheck.rules import T017_TargetLaundering, C002_TargetInFeatures
        t017 = T017_TargetLaundering()
        c002 = C002_TargetInFeatures()
        diags = []
        for node in ast.walk(tree):
            diags.extend(t017.check_node(node, None))
            diags.extend(c002.check_node(node, None))
        assert not any(d.code == "T017" for d in diags), f"Fold-local target encoding must not trigger T017: {diags}"
        assert not any(d.code == "C002" for d in diags), f"Fold-local target feature on train_df must not trigger C002: {diags}"

    def test_t017_global_target_encoding_flagged(self):
        code = """
# Global target laundering across un-split dataset
df['regime_success_rate'] = df.groupby('regime')['target'].transform('mean')
"""
        tree = ast.parse(code)
        from pycheck.rules import T017_TargetLaundering
        t017 = T017_TargetLaundering()
        diags = []
        for node in ast.walk(tree):
            diags.extend(t017.check_node(node, None))
        assert any(d.code == "T017" for d in diags), "Global groupby transform on target must trigger T017"

    def test_t017_unseen_regimes_inference_warning(self):
        code = """
regime_stats = train_df.groupby('regime')['target'].mean()
# Missing .fillna() for unseen regimes at test inference time
test_df['regime_success_rate'] = test_df['regime'].map(regime_stats)
"""
        tree = ast.parse(code)
        from pycheck.rules import T017_TargetLaundering
        t017 = T017_TargetLaundering()
        diags = []
        for node in ast.walk(tree):
            diags.extend(t017.check_node(node, None))
        assert any(d.code == "T017" and d.severity == "warning" for d in diags), "Mapping to test_df without fillna must trigger unseen regimes warning"

    def test_provenance_full_pipeline_analyze_file(self, tmp_path):
        test_file = tmp_path / "leaky_provenance.py"
        test_file.write_text("""
import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.DataFrame({'a': [1, 2]})
X_test = pd.DataFrame({'a': [3, 4]})

X_train = pd.concat([X_train, X_test])
scaler = StandardScaler()
scaler.fit_transform(X_train)
""", encoding='utf-8')
        diagnostics, _ = analyze_file(str(test_file))
        assert any(d['code'] == "F002" for d in diagnostics), f"Full analyze_file must detect F002, got: {diagnostics}"
        f002 = [d for d in diagnostics if d['code'] == "F002"][0]
        assert "X_train" in f002['message']
        assert "test partition" in f002['message']




# ═══════════════════════════════════════════════════════════════
#  STRESS TEST REGRESSION TESTS
#  From adversarial stress test: 7 false-positive fixes + 1 missed leak fix
# ═══════════════════════════════════════════════════════════════

class TestStressFalsePositives:
    """Tests for false positives identified in the adversarial stress test."""

    # C6: print(df.corr()) should be silent — EDA display
    def test_c001_silent_on_printed_corr(self):
        _check(C001_CorrelationCausation(), "print(df.corr())", expect_hit=False)

    # C6 variant: standalone expression
    def test_c001_silent_on_standalone_corr(self):
        _check(C001_CorrelationCausation(), "df.corr()", expect_hit=False)

    # C6: should still fire when assigned to a variable used for decisions
    def test_c001_fires_on_assigned_corr(self):
        _check(C001_CorrelationCausation(), "c = df['a'].corr()", expect_hit=True)

    # C7: KFold on iid data (no temporal markers) should be silent
    def test_t012_silent_on_iid_data(self):
        code = "cv = KFold(n_splits=5, shuffle=True, random_state=0)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        hits = []
        for node in ast.walk(tree):
            hits.extend(T012_KFoldTimeSeries().check_node(node, ctx))
        assert len(hits) == 0, f"T012 should not fire on non-temporal data, got: {hits}"

    # C7: KFold should still fire when temporal markers are present
    def test_t012_fires_on_temporal_data(self):
        code = "df['close'] = prices\ncv = KFold(n_splits=5, shuffle=True)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        hits = []
        for node in ast.walk(tree):
            hits.extend(T012_KFoldTimeSeries().check_node(node, ctx))
        assert len(hits) > 0, "T012 should fire on temporal data"

    # C8: train_test_split on iid images should be silent
    def test_t002_silent_on_iid_images(self):
        code = "imgs_tr, imgs_te, lab_tr, lab_te = train_test_split(images, labels, test_size=0.2, random_state=0)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        hits = []
        for node in ast.walk(tree):
            hits.extend(T002_TrainTestSplit().check_node(node, ctx))
        assert len(hits) == 0, f"T002 should not fire on iid image data, got: {hits}"

    # C9: StandardScaler inside Pipeline should be silent
    def test_t011_silent_inside_pipeline(self):
        code = "pipe = Pipeline([('s', StandardScaler()), ('m', Ridge())])"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        hits = []
        for node in ast.walk(tree):
            hits.extend(T011_ScalerBeforeSplit().check_node(node, ctx))
        assert len(hits) == 0, f"T011 should not fire inside Pipeline, got: {hits}"

    # C9: StandardScaler outside Pipeline should still fire
    def test_t011_fires_outside_pipeline(self):
        _check(T011_ScalerBeforeSplit(), "scaler = StandardScaler()", expect_hit=True)

    # C10: label = df['close'].shift(-1) is target construction, should be silent
    def test_t001_silent_on_label_construction(self):
        _check(T001_ShiftNegative(), "label = df['close'].shift(-1)", expect_hit=False)

    # C11: next_tok = tokens.shift(-1) is NLP next-token target, should be silent
    def test_t001_silent_on_next_token_target(self):
        _check(T001_ShiftNegative(), "next_tok = tokens.shift(-1)", expect_hit=False)

    # C10/C11: shift(-1) on a feature column should still fire
    def test_t001_fires_on_feature_shift(self):
        _check(T001_ShiftNegative(), "df['future_signal'] = df['close'].shift(-1)", expect_hit=True)

    # C12: expanding().mean().shift(1) is causal-safe, should be silent
    def test_t004_silent_on_expanding_with_causal_shift(self):
        code = "df['m'] = df['close'].expanding().mean().shift(1)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        hits = []
        for node in ast.walk(tree):
            hits.extend(T004_ExpandingBeforeSplit().check_node(node, ctx))
        assert len(hits) == 0, f"T004 should not fire when expanding() result has causal .shift(1), got: {hits}"

    # Expanding without shift should still fire
    def test_t004_fires_on_bare_expanding(self):
        _check(T004_ExpandingBeforeSplit(), "df['m'] = df['close'].expanding().mean()", expect_hit=True)


class TestStressMissedLeaks:
    """Tests for missed leaks identified in the adversarial stress test."""

    # B12: pct_change(periods=-1) keyword arg
    def test_t005_fires_on_keyword_periods(self):
        _check(T005_PctChangeNegative(), "df['r'] = df['close'].pct_change(periods=-1)", expect_hit=True)

    # Positive periods should still be silent
    def test_t005_silent_on_positive_keyword(self):
        _check(T005_PctChangeNegative(), "df['r'] = df['close'].pct_change(periods=5)", expect_hit=False)

class TestBoundMethodAlias:
    """B10: bound-method alias sh = df.shift; sh(-1)"""

    def test_t001_fires_on_bound_method_alias(self):
        code = "sh = df['close'].shift\ndf['future_signal'] = sh(-1)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        # Simulate the analyzer's visit_Assign to populate ctx.assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        ctx.assignments[target.id] = node.value
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                            ctx.constants[target.id] = node.value.value
                        elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                            if isinstance(node.value.operand, ast.Constant):
                                ctx.constants[target.id] = -node.value.operand.value
        hits = []
        rule = T001_ShiftNegative()
        for node in ast.walk(tree):
            hits.extend(rule.check_node(node, ctx))
        assert len(hits) > 0, "T001 should fire on bound-method alias sh(-1)"
        assert hits[0].code == "T001"

    def test_t001_silent_on_bound_method_positive(self):
        code = "sh = df['close'].shift\ndf['lag'] = sh(1)"
        tree = ast.parse(code)
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        ctx = AnalysisContext(code.split('\n'), "<test>")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        ctx.assignments[target.id] = node.value
        hits = []
        rule = T001_ShiftNegative()
        for node in ast.walk(tree):
            hits.extend(rule.check_node(node, ctx))
        assert len(hits) == 0, "T001 should not fire on bound-method alias sh(1)"


class TestNpRollNegative:
    """B4: np.roll(arr, -1) detection"""

    def test_t019_fires_on_np_roll_negative(self):
        _check(T019_NpRollNegative(), "df['y'] = np.roll(df['close'].values, -1)", expect_hit=True)

    def test_t019_silent_on_np_roll_positive(self):
        _check(T019_NpRollNegative(), "df['lag'] = np.roll(df['close'].values, 1)", expect_hit=False)

    def test_t019_fires_on_numpy_roll(self):
        _check(T019_NpRollNegative(), "arr = numpy.roll(data, -5)", expect_hit=True)

    def test_t019_silent_on_non_np_roll(self):
        _check(T019_NpRollNegative(), "result = deque.roll(-1)", expect_hit=False)

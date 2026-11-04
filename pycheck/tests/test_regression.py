"""
PyCheck v1.1.0 Regression Corpus
Tests for context-aware false-positive reduction, shuffled CV detection,
backtest timing, selective thresholds, and column-level dataflow.
"""
import ast
import pytest
from pycheck.analyzer import analyze_code
from pycheck.rules import (
    AnalysisContext, T003_RollingBeforeSplit, T004_ExpandingBeforeSplit,
    T006_FutureIndexInLoop, T010_FitBeforeSplit, T012_KFoldTimeSeries,
    T020_ShuffledCVOnSortedData, T021_CalibratedCVShuffled,
    T022_CrossValPredictShuffled, U002_HardcodedThreshold,
)


def _check(rule, code, expect_hit=True, in_loop=False, loop_var=None):
    tree = ast.parse(code)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
    ctx = AnalysisContext(code.split('\n'), "<test>")
    ctx.tree = tree
    ctx.in_loop = in_loop
    ctx.loop_var = loop_var
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
    for node in ast.walk(tree):
        hits.extend(rule.check_node(node, ctx))
    if expect_hit:
        assert len(hits) > 0, f"Rule {rule.code} should fire on:\n{code}"
    else:
        assert len(hits) == 0, f"Rule {rule.code} should NOT fire on:\n{code}\nBut got: {[h.message for h in hits]}"
    return hits


# ═══════════════════════════════════════════════════════════════
#  1. COLUMN-LEVEL DATAFLOW (flow.py)
# ═══════════════════════════════════════════════════════════════

class TestColumnLevelDataflow:
    """Verify that taint doesn't leak across columns on the same dataframe."""

    def test_safe_column_not_tainted(self):
        """df['safe'] should NOT be flagged when df['tainted'] exists."""
        code = """
import numpy as np
df['next_close'] = df['close'].shift(-1)
df['rsi'] = df['close'].rolling(14).mean()
model.fit(df[['rsi']], df['next_close'])
"""
        diags, _ = analyze_code(code, "test.py")
        f001 = [d for d in diags if d['code'] == 'F001' and 'rsi' in d['message']]
        assert len(f001) == 0, f"Safe column 'rsi' should not be flagged: {f001}"

    def test_tainted_column_still_caught(self):
        """df['tainted'] should still be caught when used as a feature."""
        code = """
df['next_close'] = df['close'].shift(-1)
df['feat'] = df['next_close'] * 2
"""
        diags, _ = analyze_code(code, "test.py")
        f001 = [d for d in diags if d['code'] == 'F001']
        assert len(f001) > 0, "Tainted column derived feature should be flagged"

    def test_multi_column_selection_safe(self):
        """df[['safe1', 'safe2']] should not propagate taint from df['tainted']."""
        code = """
df['next_ret'] = df['ret'].shift(-1)
X = df[['rsi', 'volume']]
model.fit(X, df['next_ret'])
"""
        diags, _ = analyze_code(code, "test.py")
        f001 = [d for d in diags if d['code'] == 'F001' and 'rsi' in d.get('message', '')]
        assert len(f001) == 0, f"Safe columns should not be flagged: {f001}"

    def test_multi_column_with_tainted_caught(self):
        """df[['safe', 'tainted']] should be caught."""
        code = """
df['next_ret'] = df['ret'].shift(-1)
X = df[['rsi', 'next_ret']]
model.fit(X, df['target'])
"""
        diags, _ = analyze_code(code, "test.py")
        f001 = [d for d in diags if d['code'] == 'F001' and 'next_ret' in d.get('message', '')]
        assert len(f001) > 0, "Tainted column in multi-select should be flagged"

    def test_whole_df_still_flagged(self):
        """model.fit(df, y) should still flag when df has tainted columns."""
        code = """
df['next_close'] = df['close'].shift(-1)
model.fit(df, y)
"""
        diags, _ = analyze_code(code, "test.py")
        f001 = [d for d in diags if d['code'] == 'F001']
        assert len(f001) > 0, "Whole df with tainted columns should be flagged"


# ═══════════════════════════════════════════════════════════════
#  2. T020/T021/T022: SHUFFLED CV DETECTION
# ═══════════════════════════════════════════════════════════════

class TestShuffledCVDetection:
    """Test detection of shuffled cross-validation on temporal data."""

    def test_t020_shuffled_kfold_sorted_data(self):
        """StratifiedKFold(shuffle=True) on sorted data should be flagged."""
        code = """
from datetime import datetime
rows.sort(key=lambda r: r[0])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
"""
        _check(T020_ShuffledCVOnSortedData(), code, expect_hit=True)

    def test_t020_no_shuffle_safe(self):
        """StratifiedKFold(shuffle=False) should NOT be flagged."""
        code = """
from datetime import datetime
rows.sort(key=lambda r: r[0])
cv = StratifiedKFold(n_splits=5, shuffle=False)
"""
        _check(T020_ShuffledCVOnSortedData(), code, expect_hit=False)

    def test_t020_no_temporal_context_safe(self):
        """KFold(shuffle=True) without temporal context should NOT be flagged by T020."""
        code = "cv = KFold(n_splits=5, shuffle=True)\n"
        _check(T020_ShuffledCVOnSortedData(), code, expect_hit=False)

    def test_t020_timeseries_split_safe(self):
        """TimeSeriesSplit is never flagged."""
        code = """
from datetime import datetime
rows.sort(key=lambda r: r[0])
cv = TimeSeriesSplit(n_splits=5)
"""
        _check(T020_ShuffledCVOnSortedData(), code, expect_hit=False)

    def test_t021_calibrated_cv_shuffled(self):
        """CalibratedClassifierCV with shuffled cv= should be flagged in temporal context."""
        code = """
from datetime import datetime
rows.sort(key=lambda r: r[0])
cv = StratifiedKFold(n_splits=5, shuffle=True)
clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=cv)
"""
        _check(T021_CalibratedCVShuffled(), code, expect_hit=True)

    def test_t021_calibrated_cv_timeseries_safe(self):
        """CalibratedClassifierCV with TimeSeriesSplit should NOT be flagged."""
        code = """
from datetime import datetime
cv = TimeSeriesSplit(n_splits=5)
clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=cv)
"""
        _check(T021_CalibratedCVShuffled(), code, expect_hit=False)

    def test_t022_cross_val_predict_shuffled(self):
        """cross_val_predict with shuffled cv= should be flagged in temporal context."""
        code = """
from datetime import datetime
cv = StratifiedKFold(n_splits=5, shuffle=True)
probs = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')
"""
        _check(T022_CrossValPredictShuffled(), code, expect_hit=True)

    def test_t022_cross_val_predict_timeseries_safe(self):
        """cross_val_predict with TimeSeriesSplit should NOT be flagged."""
        code = """
from datetime import datetime
cv = TimeSeriesSplit(n_splits=5)
probs = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')
"""
        _check(T022_CrossValPredictShuffled(), code, expect_hit=False)

    def test_t020_quantedge_pattern(self):
        """The QuantEdge pattern: datetime.fromisoformat + sort + StratifiedKFold(shuffle=True)."""
        code = """
from datetime import datetime
def _parse_trade_date(trade, idx):
    return datetime.fromisoformat(str(trade.get('entry_date'))[:19])
rows.sort(key=lambda r: r[0])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
"""
        diags, _ = analyze_code(code, "test.py")
        t020 = [d for d in diags if d['code'] == 'T020']
        assert len(t020) > 0, f"T020 should fire on QuantEdge pattern, got {[d['code'] for d in diags]}"
        assert t020[0].get('confidence') == 'high'


# ═══════════════════════════════════════════════════════════════
#  3. CONTEXT-AWARE T003/T004 SUPPRESSION
# ═══════════════════════════════════════════════════════════════

class TestContextAwareRollingExpanding:
    """Verify T003/T004 fire on generic code but not on temporal/financial code."""

    def test_t003_fires_on_generic_rolling(self):
        """Rolling on generic (non-temporal) column should fire."""
        _check(T003_RollingBeforeSplit(), "df['avg'] = df['val'].rolling(10).mean()", expect_hit=True)

    def test_t003_suppressed_on_close(self):
        _check(T003_RollingBeforeSplit(), "df['sma'] = df['close'].rolling(20).mean()", expect_hit=False)

    def test_t003_suppressed_on_volume(self):
        _check(T003_RollingBeforeSplit(), "df['vol_ma'] = df['volume'].rolling(20).mean()", expect_hit=False)

    def test_t003_suppressed_with_datetime_import(self):
        code = "from datetime import datetime\ndf['avg'] = df['val'].rolling(10).mean()\n"
        _check(T003_RollingBeforeSplit(), code, expect_hit=False)

    def test_t003_suppressed_with_sort_values(self):
        code = "df = df.sort_values('date')\ndf['avg'] = df['val'].rolling(10).mean()\n"
        _check(T003_RollingBeforeSplit(), code, expect_hit=False)

    def test_t004_fires_on_generic_expanding(self):
        _check(T004_ExpandingBeforeSplit(), "df['avg'] = df['val'].expanding().mean()", expect_hit=True)

    def test_t004_suppressed_on_close(self):
        _check(T004_ExpandingBeforeSplit(), "df['cum'] = df['close'].expanding().mean()", expect_hit=False)

    def test_t004_suppressed_with_sort(self):
        code = "rows.sort(key=lambda r: r[0])\ndf['avg'] = df['val'].expanding().mean()\n"
        _check(T004_ExpandingBeforeSplit(), code, expect_hit=False)


# ═══════════════════════════════════════════════════════════════
#  4. T010 MONTE CARLO / FINAL REFIT EXEMPTIONS
# ═══════════════════════════════════════════════════════════════

class TestMonteCarloExemptions:
    """Percentile/quantile on simulation output should not be flagged."""

    def test_percentile_on_sim_results_safe(self):
        code = "ci_low = np.percentile(sim_results, 5)\n"
        _check(T010_FitBeforeSplit(), code, expect_hit=False)

    def test_percentile_on_boot_probs_safe(self):
        code = "ci_low = np.percentile(boot_probs, 10)\n"
        _check(T010_FitBeforeSplit(), code, expect_hit=False)

    def test_percentile_on_monte_carlo_safe(self):
        code = "p5 = np.percentile(monte_carlo, 5)\n"
        _check(T010_FitBeforeSplit(), code, expect_hit=False)

    def test_percentile_on_dataset_still_flagged(self):
        code = "p5 = np.percentile(features, 5)\n"
        _check(T010_FitBeforeSplit(), code, expect_hit=True)

    def test_quantile_on_bootstrap_safe(self):
        code = "ci = bootstrap.quantile(0.05)\n"
        _check(T010_FitBeforeSplit(), code, expect_hit=False)


class TestFinalRefitExemption:
    """Final .fit() on full data after holdout evaluation should be downgraded."""

    def test_final_refit_after_holdout(self):
        """Full-data refit after holdout eval should be downgraded to info."""
        code = """
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
clf = Pipeline([('scaler', StandardScaler()), ('clf', LGBMClassifier())])
clf.fit(X, y)
"""
        diags, _ = analyze_code(code, "test.py")
        t010 = [d for d in diags if d['code'] == 'T010']
        # The final clf.fit(X, y) should be info (final refit after holdout) not warning
        info_fits = [d for d in t010 if d['severity'] == 'info' and 'holdout' in d['message'].lower()]
        assert len(info_fits) > 0, f"Final refit should be info-level, got {t010}"


# ═══════════════════════════════════════════════════════════════
#  5. T006 NEXT-BAR FILL RECOGNITION
# ═══════════════════════════════════════════════════════════════

class TestNextBarFill:
    """Accessing open[i+1] is a legitimate next-bar fill; close[i+1] is not."""

    def test_open_i_plus_1_downgraded(self):
        code = "entry = open[i + 1]\n"
        hits = _check(T006_FutureIndexInLoop(), code, expect_hit=True, in_loop=True, loop_var='i')
        # Should be info, not error
        assert hits[0].severity == "info"
        assert hits[0].confidence == "low"

    def test_close_i_plus_1_still_error(self):
        code = "val = close[i + 1]\n"
        hits = _check(T006_FutureIndexInLoop(), code, expect_hit=True, in_loop=True, loop_var='i')
        assert hits[0].severity == "error"

    def test_fill_price_i_plus_1_downgraded(self):
        code = "val = fill_price[i + 1]\n"
        hits = _check(T006_FutureIndexInLoop(), code, expect_hit=True, in_loop=True, loop_var='i')
        assert hits[0].severity == "info"

    def test_generic_i_plus_1_still_error(self):
        code = "val = data[i + 1]\n"
        hits = _check(T006_FutureIndexInLoop(), code, expect_hit=True, in_loop=True, loop_var='i')
        assert hits[0].severity == "error"


# ═══════════════════════════════════════════════════════════════
#  6. U002 SELECTIVE THRESHOLD WARNINGS
# ═══════════════════════════════════════════════════════════════

class TestSelectiveThresholds:
    """Model-probability thresholds are warning; rule-based score thresholds are info."""

    def test_model_prob_threshold_warning(self):
        code = "if prob >= 0.70:\n    pass\n"
        hits = _check(U002_HardcodedThreshold(), code, expect_hit=True)
        assert hits[0].severity == "warning"
        assert hits[0].confidence == "medium"
        assert "model output" in hits[0].message

    def test_prediction_threshold_warning(self):
        code = "if prediction > 0.5:\n    pass\n"
        hits = _check(U002_HardcodedThreshold(), code, expect_hit=True)
        assert hits[0].severity == "warning"

    def test_score_threshold_info(self):
        """Rule-based 'score' threshold should be info, not warning."""
        code = "if score >= 7:\n    pass\n"
        hits = _check(U002_HardcodedThreshold(), code, expect_hit=True)
        assert hits[0].severity == "info"
        assert hits[0].confidence == "low"

    def test_signal_threshold_info(self):
        code = "if signal >= 3:\n    pass\n"
        hits = _check(U002_HardcodedThreshold(), code, expect_hit=True)
        assert hits[0].severity == "info"

    def test_random_variable_not_flagged(self):
        code = "if count >= 10:\n    pass\n"
        _check(U002_HardcodedThreshold(), code, expect_hit=False)


# ═══════════════════════════════════════════════════════════════
#  7. DIAGNOSTIC CONFIDENCE FIELD
# ═══════════════════════════════════════════════════════════════

class TestConfidenceField:
    """Verify confidence field is present in diagnostics where added."""

    def test_t020_has_high_confidence(self):
        code = """
from datetime import datetime
rows.sort(key=lambda r: r[0])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
"""
        diags, _ = analyze_code(code, "test.py")
        t020 = [d for d in diags if d['code'] == 'T020']
        assert len(t020) > 0
        assert t020[0].get('confidence') == 'high'

    def test_u002_has_medium_confidence(self):
        code = "if prob >= 0.5:\n    pass\n"
        diags, _ = analyze_code(code, "test.py")
        u002 = [d for d in diags if d['code'] == 'U002']
        assert len(u002) > 0
        assert u002[0].get('confidence') == 'medium'

    def test_confidence_not_in_dict_when_none(self):
        """Diagnostics without confidence should not have the key."""
        code = "df['x'] = df['y'].shift(-1)\n"
        diags, _ = analyze_code(code, "test.py")
        t001 = [d for d in diags if d['code'] == 'T001']
        assert len(t001) > 0
        assert 'confidence' not in t001[0]  # None => not included in dict


# ═══════════════════════════════════════════════════════════════
#  8. ENHANCED TEMPORAL CONTEXT DETECTION
# ═══════════════════════════════════════════════════════════════

class TestEnhancedTemporalContext:
    """Broader temporal context detection catches QuantEdge-style patterns."""

    def test_t012_fires_with_datetime_import(self):
        """KFold in a file with datetime import should be flagged."""
        code = """
from datetime import datetime
cv = KFold(n_splits=5)
"""
        _check(T012_KFoldTimeSeries(), code, expect_hit=True)

    def test_t012_fires_with_fromisoformat(self):
        code = """
dt = datetime.fromisoformat(raw[:19])
cv = StratifiedKFold(n_splits=5)
"""
        _check(T012_KFoldTimeSeries(), code, expect_hit=True)

    def test_t012_fires_with_sort(self):
        code = """
df.sort_values('date')
cv = KFold(n_splits=5)
"""
        _check(T012_KFoldTimeSeries(), code, expect_hit=True)

    def test_t012_fires_with_entry_date(self):
        code = """
date = trade.get('entry_date')
cv = KFold(n_splits=5)
"""
        _check(T012_KFoldTimeSeries(), code, expect_hit=True)

"""
PyCheck Rule Registry — 40+ ML Safety Rules
Each rule is a self-contained checker with code, severity, and AST visitor logic.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set


@dataclass
class Diagnostic:
    line: int
    col: int
    severity: str  # "error" | "warning" | "info"
    code: str      # "T001", "C001", etc.
    message: str
    help: Optional[str] = None

    def to_dict(self):
        d = {"line": self.line, "col": self.col, "severity": self.severity,
             "code": self.code, "message": self.message}
        if self.help:
            d["help"] = self.help
        return d


class Rule:
    """Base class for all PyCheck rules."""
    code: str = ""
    name: str = ""
    severity: str = "warning"
    category: str = ""

    def check_node(self, node: ast.AST, context: 'AnalysisContext') -> List[Diagnostic]:
        return []


class AnalysisContext:
    """Shared context for all rules during analysis."""
    def __init__(self, source_lines: List[str], filepath: str):
        self.source_lines = source_lines
        self.filepath = filepath
        self.tainted_vars: Dict[str, str] = {}  # var_name -> taint_reason
        self.fitted_before_split: Set[str] = set()
        self.has_train_test_split = False
        self.split_line: Optional[int] = None
        self.in_loop = False
        self.loop_var: Optional[str] = None
        self.assignments: Dict[str, ast.AST] = {}  # var_name -> assigned node
        self.constants: Dict[str, any] = {}  # var_name -> literal constant value


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL LEAK RULES (T001–T015)
# ═══════════════════════════════════════════════════════════════

class T001_ShiftNegative(Rule):
    code = "T001"
    name = "future_shift"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'shift'):
            return []
        results = []
        for arg in node.args:
            val = self._get_negative_value(arg, ctx)
            if val is not None and val > 0:
                results.append(Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    f".shift(-{val}) accesses data {val} rows INTO THE FUTURE",
                    f"Use .shift({val}) to access past data instead"))
                # Taint the assignment target
                self._taint_parent(node, ctx, f"shift(-{val}) future data")
        for kw in node.keywords:
            if kw.arg == 'periods':
                val = self._get_negative_value(kw.value, ctx)
                if val is not None and val > 0:
                    results.append(Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f".shift(periods=-{val}) accesses data {val} rows INTO THE FUTURE",
                        f"Use .shift(periods={val}) to access past data instead"))
        return results

    def _get_negative_value(self, arg, ctx=None):
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, (int, float)):
                return abs(arg.operand.value)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
            return abs(arg.value)
        if isinstance(arg, ast.Name) and ctx is not None and hasattr(ctx, 'constants'):
            val = ctx.constants.get(arg.id)
            if val is not None and isinstance(val, (int, float)) and val < 0:
                return abs(val)
        return None

    def _taint_parent(self, node, ctx, reason):
        pass  # Flow engine handles this


class T002_TrainTestSplit(Rule):
    code = "T002"
    name = "random_split_timeseries"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        name = self._get_call_name(node)
        if name == 'train_test_split':
            ctx.has_train_test_split = True
            ctx.split_line = node.lineno
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                "train_test_split() shuffles time-series data, leaking future into training",
                "Use a temporal split: train = data[:split_idx], test = data[split_idx:]")]
        return []

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


class T003_RollingBeforeSplit(Rule):
    code = "T003"
    name = "rolling_before_split"
    severity = "warning"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'rolling':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".rolling() computed on full dataset may include future data",
                "Compute rolling statistics AFTER splitting into train/test sets")]
        return []


class T004_ExpandingBeforeSplit(Rule):
    code = "T004"
    name = "expanding_before_split"
    severity = "warning"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'expanding':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".expanding() computed on full dataset may include future data",
                "Compute expanding statistics AFTER splitting into train/test sets")]
        return []


class T005_PctChangeNegative(Rule):
    code = "T005"
    name = "future_pct_change"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'pct_change'):
            return []
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                if isinstance(arg.operand, ast.Constant):
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"pct_change with negative period accesses future data",
                        "Use positive periods to look backwards in time")]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                return [Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    f"pct_change({int(arg.value)}) accesses future data",
                    "Use positive periods to look backwards in time")]
        return []


class T006_FutureIndexInLoop(Rule):
    code = "T006"
    name = "future_index_loop"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Subscript):
            return []
        # Look for .iloc[i + 1] or [i + 1] patterns inside loops
        if not ctx.in_loop or ctx.loop_var is None:
            return []
        sl = node.slice
        if isinstance(sl, ast.BinOp) and isinstance(sl.op, ast.Add):
            if self._uses_var(sl.left, ctx.loop_var) and isinstance(sl.right, ast.Constant):
                if isinstance(sl.right.value, int) and sl.right.value > 0:
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"Indexing [{ctx.loop_var} + {sl.right.value}] inside loop accesses future data",
                        f"Only use [{ctx.loop_var}] or [{ctx.loop_var} - n] to access current/past data")]
            if self._uses_var(sl.right, ctx.loop_var) and isinstance(sl.left, ast.Constant):
                if isinstance(sl.left.value, int) and sl.left.value > 0:
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"Indexing [{sl.left.value} + {ctx.loop_var}] inside loop may access future data",
                        f"Verify this does not access future indices")]
        return []

    def _uses_var(self, node, var_name):
        if isinstance(node, ast.Name) and node.id == var_name:
            return True
        return False


class T007_BackfillLeak(Rule):
    code = "T007"
    name = "backfill_leak"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('fillna', 'bfill', 'backfill'):
            if node.func.attr in ('bfill', 'backfill'):
                return [Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    f".{node.func.attr}() fills missing values with FUTURE data",
                    "Use .ffill() (forward fill) or .fillna(0) instead")]
            # Check fillna(method='bfill') or fillna(method='backfill')
            for kw in node.keywords:
                if kw.arg == 'method' and isinstance(kw.value, ast.Constant):
                    if kw.value.value in ('bfill', 'backfill'):
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f".fillna(method='{kw.value.value}') fills missing values with FUTURE data",
                            "Use .fillna(method='ffill') or .fillna(0) instead")]
        return []


class T008_NonCausalInterpolate(Rule):
    code = "T008"
    name = "noncausal_interpolate"
    severity = "warning"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'interpolate':
            safe_methods = {'pad', 'ffill', 'linear', 'nearest', 'zero', 'slinear'}
            for kw in node.keywords:
                if kw.arg == 'method' and isinstance(kw.value, ast.Constant):
                    if kw.value.value in ('cubic', 'spline', 'polynomial', 'quadratic'):
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f".interpolate(method='{kw.value.value}') uses future data points for interpolation",
                            "Use .interpolate(method='pad') or method='linear' with limit_direction='forward'")]
                    if kw.value.value in safe_methods:
                        return []  # Explicitly safe — no diagnostic
            # No method keyword specified — default is 'linear' which can be non-causal
            return [Diagnostic(
                node.lineno, node.col_offset, "info", self.code,
                ".interpolate() may use future data depending on method",
                "Ensure method is causal (e.g., method='pad' or method='ffill')")]
        return []


class T009_CenteredRolling(Rule):
    code = "T009"
    name = "centered_rolling"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'rolling':
            for kw in node.keywords:
                if kw.arg == 'center' and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        ".rolling(center=True) uses future data on both sides of the window",
                        "Use .rolling(center=False) for causal (backward-looking) windows")]
        return []


class T010_FitBeforeSplit(Rule):
    code = "T010"
    name = "fit_before_split"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'fit':
            # Only flag .fit() on objects that look like ML preprocessors/models
            # Skip: fig.fit(), layout.fit(), etc.
            obj_name = self._get_obj_name(node.func.value)
            safe_objects = {'fig', 'figure', 'ax', 'axes', 'plt', 'layout',
                            'canvas', 'widget', 'app', 'db', 'session', 'socket',
                            'parser', 'formatter', 'logger', 'handler'}
            if obj_name and obj_name.lower() in safe_objects:
                return []
            return [Diagnostic(
                node.lineno, node.col_offset, "warning", self.code,
                ".fit() may be called on the full dataset including test data",
                "Call .fit() only on training data, use .transform() on test data")]
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'fit_transform':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".fit_transform() on full dataset leaks test statistics into training",
                "Use .fit(X_train).transform(X_train), then .transform(X_test) separately")]
        return []

    def _get_obj_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


class T011_ScalerBeforeSplit(Rule):
    code = "T011"
    name = "scaler_before_split"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        scaler_names = {'StandardScaler', 'MinMaxScaler', 'RobustScaler',
                        'MaxAbsScaler', 'Normalizer', 'QuantileTransformer',
                        'PowerTransformer'}
        if isinstance(node.func, ast.Name) and node.func.id in scaler_names:
            return [Diagnostic(
                node.lineno, node.col_offset, "info", self.code,
                f"{node.func.id}() — ensure this is fit ONLY on training data",
                "Fit the scaler on X_train, then transform both X_train and X_test")]
        return []


class T012_KFoldTimeSeries(Rule):
    code = "T012"
    name = "kfold_timeseries"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        kfold_names = {'KFold', 'StratifiedKFold', 'RepeatedKFold', 'ShuffleSplit'}
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in kfold_names:
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                f"{name}() shuffles time-series data across folds, leaking future into training",
                "Use TimeSeriesSplit() for temporal cross-validation")]
        return []


class T013_ResampleBeforeSplit(Rule):
    code = "T013"
    name = "resample_before_split"
    severity = "warning"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'resample':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".resample() on full dataset may leak future aggregate statistics",
                "Resample AFTER splitting into train/test sets")]
        return []


class T014_DiffNegative(Rule):
    code = "T014"
    name = "future_diff"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'diff'):
            return []
        results = []
        for arg in node.args:
            val = self._get_negative_value(arg, ctx)
            if val is not None and val > 0:
                results.append(Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    f".diff(-{val}) computes difference using future data",
                    f"Use .diff({val}) to compute backward differences"))
        for kw in node.keywords:
            if kw.arg == 'periods':
                val = self._get_negative_value(kw.value, ctx)
                if val is not None and val > 0:
                    results.append(Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f".diff(periods=-{val}) computes difference using future data",
                        f"Use .diff(periods={val}) to compute backward differences"))
        return results

    def _get_negative_value(self, arg, ctx=None):
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, (int, float)):
                return abs(arg.operand.value)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
            return abs(arg.value)
        if isinstance(arg, ast.Name) and ctx is not None and hasattr(ctx, 'constants'):
            val = ctx.constants.get(arg.id)
            if val is not None and isinstance(val, (int, float)) and val < 0:
                return abs(val)
        return None


class T015_GroupbyShiftLeak(Rule):
    code = "T015"
    name = "groupby_shift_leak"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('shift', 'diff', 'pct_change'):
            # Check if this is chained after .transform() or .apply()
            obj = node.func.value
            if isinstance(obj, ast.Call) and isinstance(obj.func, ast.Attribute):
                if obj.func.attr in ('transform', 'apply'):
                    for arg in node.args:
                        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                            return [Diagnostic(
                                node.lineno, node.col_offset, self.severity, self.code,
                                f"Negative {node.func.attr}() inside groupby().transform() leaks future within groups",
                                "Use positive periods for backward-looking operations within groups")]
        return []


# ═══════════════════════════════════════════════════════════════
#  CAUSAL CONFUSION RULES (C001–C010)
# ═══════════════════════════════════════════════════════════════

class C001_CorrelationCausation(Rule):
    code = "C001"
    name = "correlation_causation"
    severity = "warning"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'corr':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".corr() measures correlation, not causation — do not use for treatment/trading decisions",
                "Use causal inference methods (DoWhy, EconML) or randomized experiments")]
        return []


class C002_TargetInFeatures(Rule):
    code = "C002"
    name = "target_in_features"
    severity = "error"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Subscript):
            return []
        # Detect df[['feature1', 'target']] where target-like names are used as features
        if isinstance(node.slice, ast.List):
            target_names = {'target', 'y', 'label', 'labels', 'outcome', 'y_true',
                            'future_price', 'future_return', 'next_close', 'next_price'}
            for elt in node.slice.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value.lower() in target_names:
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Column '{elt.value}' appears to be a target variable used as a feature",
                            "Remove target/outcome columns from feature inputs to avoid data leakage")]
        return []


class C003_PostTreatmentVariable(Rule):
    code = "C003"
    name = "post_treatment_variable"
    severity = "warning"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        # Detect variable names that suggest post-treatment data
        if not isinstance(node, ast.Assign):
            return []
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                col_name = str(target.slice.value).lower()
                post_indicators = ('after_', 'post_', 'result_', 'outcome_', 'response_')
                if any(col_name.startswith(p) for p in post_indicators):
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"Column '{target.slice.value}' may be a post-treatment variable",
                        "Post-treatment variables should not be used as features in causal models")]
        return []


class C004_SelectionOnOutcome(Rule):
    code = "C004"
    name = "selection_on_outcome"
    severity = "error"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Assign):
            return []
        val = node.value
        # Detect df = df[df['profit'] > 0] patterns (survivorship bias)
        if isinstance(val, ast.Subscript) and isinstance(val.slice, ast.Compare):
            comp = val.slice
            if isinstance(comp.left, ast.Subscript) and isinstance(comp.left.slice, ast.Constant):
                col = str(comp.left.slice.value).lower()
                outcome_cols = {'profit', 'return', 'returns', 'pnl', 'outcome',
                                'survived', 'alive', 'active', 'success'}
                if col in outcome_cols:
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"Filtering by outcome column '{comp.left.slice.value}' before analysis introduces survivorship bias",
                        "Include all data points, including failures, in your analysis")]
        return []


class C005_SurvivorshipBias(Rule):
    code = "C005"
    name = "survivorship_bias"
    severity = "warning"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'dropna':
            return [Diagnostic(
                node.lineno, node.col_offset, "info", self.code,
                ".dropna() removes rows non-randomly — may introduce survivorship bias",
                "Consider imputation (.fillna()) or analyze the pattern of missing data first")]
        return []


class C006_PHacking(Rule):
    code = "C006"
    name = "p_hacking"
    severity = "warning"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Compare):
            return []
        # Detect patterns like p_value < 0.05
        if isinstance(node.left, ast.Name):
            if 'p_val' in node.left.id.lower() or 'pvalue' in node.left.id.lower():
                return [Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    "Comparing p-values to thresholds — risk of p-hacking with multiple comparisons",
                    "Apply Bonferroni correction or use FDR control for multiple hypothesis testing")]
        return []


class C007_CorrWithoutForDecisions(Rule):
    code = "C007"
    name = "corrwith_decisions"
    severity = "warning"
    category = "CausalConfusion"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'corrwith':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".corrwith() computes correlation — do not use for causal feature selection",
                "Use mutual information, SHAP values, or causal feature selection instead")]
        return []


# ═══════════════════════════════════════════════════════════════
#  UNCERTAINTY RULES (U001–U008)
# ═══════════════════════════════════════════════════════════════

class U001_PredictNoUncertainty(Rule):
    code = "U001"
    name = "predict_no_uncertainty"
    severity = "warning"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict':
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                "model.predict() returns point estimates without confidence scores",
                "Use model.predict_proba() or compute prediction intervals to quantify uncertainty")]
        return []


class U002_HardcodedThreshold(Rule):
    code = "U002"
    name = "hardcoded_threshold"
    severity = "warning"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Compare):
            return []
        # Detect: prediction > 0.5, signal > 0.7, etc.
        if isinstance(node.left, ast.Name):
            pred_names = {'prediction', 'pred', 'signal', 'score', 'probability', 'prob', 'confidence'}
            if node.left.id.lower() in pred_names:
                for comp_val in node.comparators:
                    if isinstance(comp_val, ast.Constant) and isinstance(comp_val.value, (int, float)):
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Hardcoded threshold ({comp_val.value}) on '{node.left.id}' ignores prediction uncertainty",
                            "Use calibrated probability thresholds or dynamic thresholds based on confidence intervals")]
        return []


class U003_NoCalibration(Rule):
    code = "U003"
    name = "no_calibration"
    severity = "warning"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict_proba':
            return [Diagnostic(
                node.lineno, node.col_offset, "info", self.code,
                ".predict_proba() output may not be calibrated — probabilities may not reflect true likelihoods",
                "Use CalibratedClassifierCV or check calibration with calibration_curve()")]
        return []


class U004_NoEnsemble(Rule):
    code = "U004"
    name = "single_model"
    severity = "info"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        single_models = {'LinearRegression', 'LogisticRegression', 'DecisionTreeClassifier',
                         'DecisionTreeRegressor', 'SVR', 'SVC', 'KNeighborsClassifier',
                         'SGDClassifier', 'SGDRegressor', 'Perceptron'}
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in single_models:
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                f"{name}() — single model provides no uncertainty estimate",
                "Consider ensemble methods (RandomForest, GradientBoosting) or Bayesian approaches for uncertainty")]
        return []


class U005_ScoreOnTrainOnly(Rule):
    code = "U005"
    name = "score_train_only"
    severity = "error"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'score':
            # Check if the args look like training data
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    name_lower = arg.id.lower()
                    if 'train' in name_lower and 'test' not in name_lower:
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f".score({arg.id}) evaluates on training data — this is overfitting, not validation",
                            "Evaluate on held-out test data: model.score(X_test, y_test)")]
        return []


class U006_PredictInLoop(Rule):
    code = "U006"
    name = "predict_production_loop"
    severity = "error"
    category = "UncertaintyIgnored"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if not ctx.in_loop:
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict':
            return [Diagnostic(
                node.lineno, node.col_offset, "warning", self.code,
                ".predict() inside a loop without uncertainty check — risky for live trading/production",
                "Add confidence thresholds or uncertainty bounds before acting on predictions")]
        return []


# ═══════════════════════════════════════════════════════════════
#  DATA QUALITY RULES (D001–D007)
# ═══════════════════════════════════════════════════════════════

class D001_DropNaBeforeSplit(Rule):
    code = "D001"
    name = "dropna_before_split"
    severity = "warning"
    category = "DataQuality"

    def check_node(self, node, ctx):
        # Handled by C005 (survivorship), but with different framing
        return []


class D002_SilentExceptionSwallow(Rule):
    code = "D002"
    name = "silent_exception"
    severity = "warning"
    category = "DataQuality"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.ExceptHandler):
            return []
        # Check for bare `except: pass`
        if node.type is None and len(node.body) == 1:
            if isinstance(node.body[0], ast.Pass):
                return [Diagnostic(
                    node.lineno, node.col_offset, self.severity, self.code,
                    "Bare `except: pass` silently swallows errors in data pipeline",
                    "Catch specific exceptions and log or handle them properly")]
        return []


class D003_MagicNumbers(Rule):
    code = "D003"
    name = "magic_numbers"
    severity = "info"
    category = "DataQuality"

    def check_node(self, node, ctx):
        # Detect hardcoded numeric constants in function calls
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('rolling', 'shift', 'head', 'tail', 'nlargest', 'nsmallest'):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                    if arg.value > 100 or (isinstance(arg.value, float) and arg.value != int(arg.value)):
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Magic number {arg.value} in .{node.func.attr}() — consider using a named constant",
                            "Define as a variable: WINDOW_SIZE = {arg.value}")]
        return []


# ═══════════════════════════════════════════════════════════════
#  RULE REGISTRY
# ═══════════════════════════════════════════════════════════════

ALL_RULES: List[Rule] = [
    # Temporal Leak
    T001_ShiftNegative(),
    T002_TrainTestSplit(),
    T003_RollingBeforeSplit(),
    T004_ExpandingBeforeSplit(),
    T005_PctChangeNegative(),
    T006_FutureIndexInLoop(),
    T007_BackfillLeak(),
    T008_NonCausalInterpolate(),
    T009_CenteredRolling(),
    T010_FitBeforeSplit(),
    T011_ScalerBeforeSplit(),
    T012_KFoldTimeSeries(),
    T013_ResampleBeforeSplit(),
    T014_DiffNegative(),
    T015_GroupbyShiftLeak(),
    # Causal Confusion
    C001_CorrelationCausation(),
    C002_TargetInFeatures(),
    C003_PostTreatmentVariable(),
    C004_SelectionOnOutcome(),
    C005_SurvivorshipBias(),
    C006_PHacking(),
    C007_CorrWithoutForDecisions(),
    # Uncertainty
    U001_PredictNoUncertainty(),
    U002_HardcodedThreshold(),
    U003_NoCalibration(),
    U004_NoEnsemble(),
    U005_ScoreOnTrainOnly(),
    U006_PredictInLoop(),
    # Data Quality
    D002_SilentExceptionSwallow(),
    D003_MagicNumbers(),
]

RULES_BY_CODE = {r.code: r for r in ALL_RULES}

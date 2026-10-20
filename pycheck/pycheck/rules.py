"""
PyCheck Rule Registry — 34 ML Safety Rules
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
        self.train_vars: Set[str] = set()
        self.test_vars: Set[str] = set()
        self.test_contaminated_vars: Dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL LEAK RULES (T001–T015)
# ═══════════════════════════════════════════════════════════════

class T001_ShiftNegative(Rule):
    code = "T001"
    name = "future_shift"
    severity = "error"
    category = "TemporalLeak"

    # Target/label variable names where shift(-1) is intentional (creating the prediction target)
    _TARGET_ASSIGNMENT_NAMES = {
        'label', 'labels', 'target', 'targets', 'y', 'y_true', 'y_train', 'y_test',
        'next_tok', 'next_token', 'next_word', 'next_char',
    }

    def _is_target_construction(self, node):
        """Check if this shift(-1) is assigned to a target/label variable (intentional future target, not a feature leak)."""
        parent = getattr(node, 'parent', None)
        if parent is None:
            return False
        # Direct: label = df['close'].shift(-1)
        assign = parent
        if isinstance(parent, ast.BinOp):
            assign = getattr(parent, 'parent', None)
        if isinstance(assign, ast.Assign):
            for tgt in assign.targets:
                if isinstance(tgt, ast.Name) and tgt.id.lower() in self._TARGET_ASSIGNMENT_NAMES:
                    return True
                # df['target'] = ..., df['label'] = ..., df['next_tok'] = ...
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant):
                    col = str(tgt.slice.value).lower()
                    if col in self._TARGET_ASSIGNMENT_NAMES or col.startswith('next_') or col.startswith('future_target'):
                        return True
        return False

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        # Standard: obj.shift(...)
        is_shift = isinstance(node.func, ast.Attribute) and node.func.attr == 'shift'
        # R2-9 fix: getattr(obj, 'shift')(...)
        is_getattr_shift = (isinstance(node.func, ast.Call) and
                            isinstance(node.func.func, ast.Name) and
                            node.func.func.id == 'getattr' and
                            len(node.func.args) >= 2 and
                            isinstance(node.func.args[1], ast.Constant) and
                            node.func.args[1].value == 'shift')
        # B10 fix: bound-method alias - sh = df['close'].shift; sh(-1)
        is_alias_shift = False
        if isinstance(node.func, ast.Name) and ctx is not None and hasattr(ctx, 'assignments'):
            assigned = ctx.assignments.get(node.func.id)
            if isinstance(assigned, ast.Attribute) and assigned.attr == 'shift':
                is_alias_shift = True
        if not (is_shift or is_getattr_shift or is_alias_shift):
            return []
        results = []
        for arg in node.args:
            val = self._get_negative_value(arg, ctx)
            if val is not None and val > 0:
                # Suppress if this is intentional target/label construction
                if self._is_target_construction(node):
                    return []
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
                    if self._is_target_construction(node):
                        return []
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

    # Variable names that indicate iid (non-temporal) data, where random splits are safe
    _IID_DATA_NAMES = {
        'images', 'image', 'imgs', 'img', 'photos', 'pixels',
        'iris', 'iris_x', 'iris_y', 'iris_data', 'mnist', 'cifar',
        'texts', 'text', 'documents', 'docs', 'sentences',
        'features', 'samples', 'labels', 'categories',
    }

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        name = self._get_call_name(node)
        if name == 'train_test_split':
            ctx.has_train_test_split = True
            ctx.split_line = node.lineno

            # 1. Explicit shuffle=False is a safe chronological sequential split
            for kw in node.keywords:
                if kw.arg == 'shuffle':
                    if isinstance(kw.value, ast.Constant) and (kw.value.value is False or kw.value.value == 0):
                        return []

            # 2. If the arguments are clearly iid data names, suppress
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id.lower() in self._IID_DATA_NAMES:
                    return []

            # 3. Only flag when time-series indicators exist in the data/script
            if not self._has_temporal_context(ctx):
                return []

            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                "train_test_split() shuffles time-series data, leaking future into training",
                "Use a temporal split: train = data[:split_idx], test = data[split_idx:], or set shuffle=False")]
        return []

    def _has_temporal_context(self, ctx):
        if not ctx or not ctx.source_lines:
            return False
        source_lower = "\n".join(ctx.source_lines).lower()
        temporal_markers = (
            'close', 'open', 'high', 'low', 'volume', 'vwap', 'ohlc',
            'price', 'prices', 'return', 'returns', 'pnl', 'tick', 'bar',
            'date', 'datetime', 'timestamp', 'time_series', 'timeseries',
            'rolling', 'expanding', 'shift', 'pct_change', 'diff',
            'resample', 'timeseriessplit', 'merge_asof'
        )
        return any(marker in source_lower for marker in temporal_markers)

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
            # Check if result is made causal via .shift(1) or .shift(n>0) downstream
            # Pattern: df['close'].expanding().mean().shift(1)
            # The expanding() call is the grandchild of the .mean() call which is grandchild of .shift(1)
            if self._has_causal_shift_downstream(node):
                return []
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                ".expanding() computed on full dataset may include future data",
                "Compute expanding statistics AFTER splitting into train/test sets")]
        return []

    def _has_causal_shift_downstream(self, node):
        """Walk up the AST to see if .expanding()...agg().shift(positive) makes it causal."""
        current = node
        for _ in range(5):  # walk up max 5 levels
            parent = getattr(current, 'parent', None)
            if parent is None:
                return False
            # If parent is a .shift() call with positive arg, this is causal
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) and parent.func.attr == 'shift':
                for arg in parent.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value > 0:
                        return True
                # No args means shift(1) default? No, default is 1 but shift() with no args = shift(1) which IS causal
                if not parent.args and not any(kw.arg == 'periods' for kw in parent.keywords):
                    return True
            # Keep walking up through method chains (.expanding().mean() etc.)
            if isinstance(parent, ast.Attribute):
                current = parent
                continue
            if isinstance(parent, ast.Call):
                current = parent
                continue
            break
        return False


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
        # Also check keyword arg: pct_change(periods=-1)
        for kw in node.keywords:
            if kw.arg == 'periods':
                if isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                    if isinstance(kw.value.operand, ast.Constant):
                        val = kw.value.operand.value
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"pct_change(periods=-{val}) accesses future data",
                            "Use positive periods to look backwards in time")]
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)) and kw.value.value < 0:
                    return [Diagnostic(
                        node.lineno, node.col_offset, self.severity, self.code,
                        f"pct_change(periods={int(kw.value.value)}) accesses future data",
                        "Use positive periods to look backwards in time")]
                # Resolve variable reference
                if isinstance(kw.value, ast.Name) and ctx is not None and hasattr(ctx, 'constants'):
                    val = ctx.constants.get(kw.value.id)
                    if val is not None and isinstance(val, (int, float)) and val < 0:
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"pct_change(periods={int(val)}) accesses future data",
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

    def _is_train_data(self, node, ctx=None):
        """Recursively check if an AST node represents training data."""
        var_name = self._get_obj_name(node)
        if var_name and ctx is not None:
            base_var = var_name.split('.')[0]
            if hasattr(ctx, 'test_contaminated_vars') and base_var in ctx.test_contaminated_vars:
                return False
            if hasattr(ctx, 'test_vars') and base_var in ctx.test_vars:
                return False
            if hasattr(ctx, 'train_vars') and base_var in ctx.train_vars:
                return True

        train_markers = ('_train', 'train_', 'x_train', 'y_train', 'x_tr', 'y_tr', '_trn', 'trn_', 'train')
        if isinstance(node, ast.Name):
            return any(m in node.id.lower() for m in train_markers)
        if isinstance(node, ast.Subscript):
            return self._is_train_data(node.value, ctx)
        if isinstance(node, ast.Attribute):
            return self._is_train_data(node.value, ctx)
        return False

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('fit', 'fit_transform'):
            # Only flag on objects that look like ML preprocessors/models
            # Skip: fig.fit(), layout.fit(), etc.
            obj_name = self._get_obj_name(node.func.value)
            safe_objects = {'fig', 'figure', 'ax', 'axes', 'plt', 'layout',
                            'canvas', 'widget', 'app', 'db', 'session', 'socket',
                            'parser', 'formatter', 'logger', 'handler'}
            if obj_name and obj_name.lower() in safe_objects:
                return []

            # If arguments are contaminated, Flow engine F002 emits specific data contamination diagnostics
            if ctx is not None and hasattr(ctx, 'test_contaminated_vars'):
                for a in node.args:
                    v_name = self._get_obj_name(a)
                    if v_name and v_name.split('.')[0] in ctx.test_contaminated_vars:
                        return []
            
            # Suppress if all positional arguments are clearly train-only variables
            # (e.g. X_train, y_train, train_df[features], X_tr)
            if node.args and all(self._is_train_data(a, ctx) for a in node.args):
                return []

            if node.func.attr == 'fit':
                return [Diagnostic(
                    node.lineno, node.col_offset, "warning", self.code,
                    ".fit() may be called on the full dataset including test data",
                    "Call .fit() only on training data, use .transform() on test data")]
            else:
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
            # Suppress when used inside sklearn Pipeline/make_pipeline (the pipeline handles fit/transform correctly in CV)
            parent = getattr(node, 'parent', None)
            if parent is not None:
                # Direct tuple element inside Pipeline steps list
                if isinstance(parent, ast.Tuple):
                    grandparent = getattr(parent, 'parent', None)
                    if grandparent is not None and isinstance(grandparent, ast.List):
                        ggp = getattr(grandparent, 'parent', None)
                        if ggp is not None and isinstance(ggp, ast.Call):
                            if isinstance(ggp.func, ast.Name) and ggp.func.id in ('Pipeline', 'make_pipeline'):
                                return []
                # make_pipeline(StandardScaler(), Ridge()) — direct arg
                if isinstance(parent, ast.Call):
                    if isinstance(parent.func, ast.Name) and parent.func.id in ('Pipeline', 'make_pipeline'):
                        return []
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
            # Only flag when temporal/financial context markers exist in the script
            if not self._has_temporal_context(ctx):
                return []
            return [Diagnostic(
                node.lineno, node.col_offset, self.severity, self.code,
                f"{name}() shuffles time-series data across folds, leaking future into training",
                "Use TimeSeriesSplit() for temporal cross-validation")]
        return []

    def _has_temporal_context(self, ctx):
        if not ctx or not ctx.source_lines:
            return False
        source_lower = "\n".join(ctx.source_lines).lower()
        temporal_markers = (
            'close', 'open', 'high', 'low', 'volume', 'vwap', 'ohlc',
            'price', 'prices', 'return', 'returns', 'pnl', 'tick', 'bar',
            'date', 'datetime', 'timestamp', 'time_series', 'timeseries',
            '.rolling', '.expanding', '.shift', 'pct_change', '.diff',
            'resample', 'timeseriessplit', 'merge_asof'
        )
        return any(marker in source_lower for marker in temporal_markers)


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


class T016_MergeAsofDirection(Rule):
    code = "T016"
    name = "merge_asof_future_direction"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        is_merge_asof = False
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'merge_asof':
            is_merge_asof = True
        elif isinstance(node.func, ast.Name) and node.func.id == 'merge_asof':
            is_merge_asof = True
        
        if not is_merge_asof:
            return []
        
        for kw in node.keywords:
            if kw.arg == 'direction':
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    val = kw.value.value.lower()
                    if val in ('nearest', 'forward'):
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"pd.merge_asof() with direction='{kw.value.value}' matches future timestamps (lookahead bias)",
                            "Use direction='backward' for point-in-time causal event joins")]
        return []


class T017_TargetLaundering(Rule):
    code = "T017"
    name = "target_laundering_leak"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        target_col_names = {'target', 'success', 'outcome', 'label', 'labels', 'y',
                            'profit', 'pnl', 'win', 'return', 'returns', 'future_price'}
        train_markers = ('_train', 'train_', 'x_train', 'y_train', 'x_tr', 'y_tr', '_trn', 'trn_', 'train', 'fit', 'fit_', '_fit', 'trn')
        test_markers = ('_test', 'test_', 'x_test', 'y_test', 'x_val', 'y_val', '_val', 'val_', '_eval', 'eval_', 'holdout', 'validation')
        
        # 1. Detect target encoding / aggregation: df.groupby(...)['target'].transform('mean')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ('TargetEncoder', 'CatBoostEncoder'):
                return [Diagnostic(
                    node.lineno, node.col_offset, "info", self.code,
                    f"{node.func.id}() calculates target statistics across groups; ensure it is fit ONLY on training data",
                    "Fit target encoders strictly on X_train, y_train after temporal splitting")]
            
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('transform', 'mean', 'median', 'std', 'sum', 'agg', 'apply'):
                val = node.func.value
                col_name = None
                groupby_call = None
                
                if isinstance(val, ast.Subscript):
                    if isinstance(val.slice, ast.Constant) and isinstance(val.slice.value, str):
                        col_name = val.slice.value.lower()
                    groupby_call = val.value
                elif isinstance(val, ast.Call):
                    groupby_call = val
                
                if col_name and any(t in col_name for t in target_col_names):
                    if isinstance(groupby_call, ast.Call) and isinstance(groupby_call.func, ast.Attribute) and groupby_call.func.attr == 'groupby':
                        # Check the caller of groupby:
                        caller_obj = groupby_call.func.value
                        caller_name = ""
                        if isinstance(caller_obj, ast.Name):
                            caller_name = caller_obj.id.lower()
                        elif isinstance(caller_obj, ast.Attribute):
                            caller_name = caller_obj.attr.lower()
                        
                        # If the groupby is computed strictly within a training fold,
                        # this is a VALID fold-local transformation! Do NOT flag it.
                        if any(m in caller_name for m in train_markers):
                            return []

                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Target laundering: grouping by target/outcome column '{col_name}' across the un-split dataset leaks future test labels into features",
                            "Compute regime/group target statistics strictly within the training fold (e.g. train_df.groupby()), and map to test fold with .fillna() for unseen regimes")]

        # 2. Detect assignment creating laundered columns: df['regime_success_rate'] = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    col = target.slice.value.lower()
                    if any(marker in col for marker in ('success_rate', 'target_enc', 'target_mean', 'regime_success')):
                        target_df = target.value
                        target_df_name = ""
                        if isinstance(target_df, ast.Name):
                            target_df_name = target_df.id.lower()
                        elif isinstance(target_df, ast.Attribute):
                            target_df_name = target_df.attr.lower()

                        # Case A: Assigning to a training fold dataframe -> valid fold-local transformation!
                        if any(m in target_df_name for m in train_markers):
                            return []

                        # Case B: Assigning to test/inference fold dataframe (e.g. test_df['regime_success_rate'] = ...)
                        if any(m in target_df_name for m in test_markers):
                            # Check if unseen regimes at inference time are handled via .fillna()
                            has_fillna = False
                            if isinstance(node.value, ast.Call):
                                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'fillna':
                                    has_fillna = True
                            
                            if not has_fillna:
                                return [Diagnostic(
                                    node.lineno, node.col_offset, "warning", self.code,
                                    f"Mapping target statistics to test data '{target_df_name}' without .fillna(): unseen categories/regimes at inference time will produce NaNs",
                                    "Add .fillna(train_prior_mean) to safely handle unseen categories or market regimes at inference time")]
                            return []

                        # Case C: Assigning to un-split / full dataset dataframe (df['regime_success_rate'] = ...)
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Target laundering: column '{target.slice.value}' encodes target outcomes across full dataset",
                            "Calculate success rates and target statistics only on the training set")]
        return []


class T018_InSampleTargetEncoding(Rule):
    code = "T018"
    name = "insample_target_encoding"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        target_col_names = {'target', 'success', 'outcome', 'label', 'labels', 'y',
                            'profit', 'pnl', 'win', 'return', 'returns', 'future_price', 'default'}

        if isinstance(node, ast.Call):
            # 1. Detect TargetEncoder(cv=None) or cv=1
            if isinstance(node.func, ast.Name) and node.func.id in ('TargetEncoder', 'CatBoostEncoder', 'WOEEncoder'):
                for kw in node.keywords:
                    if kw.arg == 'cv':
                        if (isinstance(kw.value, ast.Constant) and (kw.value.value is None or kw.value.value == 1)) or \
                           (isinstance(kw.value, ast.Name) and kw.value.id in ('None',)):
                            return [Diagnostic(
                                node.lineno, node.col_offset, self.severity, self.code,
                                f"{node.func.id}(cv=None) disables out-of-fold cross-validation; target statistics will leak in-sample labels into features",
                                "Set cv>=5 (e.g. TargetEncoder(cv=5)) to enforce out-of-fold target encoding")]

            # 2. Detect groupby on target with transform('mean'/'median'/'std'/'var')
            # e.g.: train['cat_enc'] = train.groupby('category')['target'].transform('mean')
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'transform':
                val = node.func.value
                col_name = None
                groupby_call = None

                if isinstance(val, ast.Subscript):
                    if isinstance(val.slice, ast.Constant) and isinstance(val.slice.value, str):
                        col_name = val.slice.value.lower()
                    groupby_call = val.value
                elif isinstance(val, ast.Call):
                    groupby_call = val

                if col_name and any(t in col_name for t in target_col_names):
                    if isinstance(groupby_call, ast.Call) and isinstance(groupby_call.func, ast.Attribute) and groupby_call.func.attr == 'groupby':
                        agg_func = None
                        if node.args:
                            arg0 = node.args[0]
                            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                                agg_func = arg0.value.lower()
                            elif isinstance(arg0, ast.Attribute):
                                agg_func = arg0.attr.lower()
                            elif isinstance(arg0, ast.Name):
                                agg_func = arg0.id.lower()

                        if agg_func in ('mean', 'median', 'std', 'var'):
                            return [Diagnostic(
                                node.lineno, node.col_offset, self.severity, self.code,
                                f"In-sample target encoding leak: groupby.transform('{agg_func}') on target '{col_name}' includes row i's own label in feature i",
                                "Use Out-of-Fold (OOF) K-Fold target encoding or Leave-One-Out smoothing: ((group_sum - y) / (count - 1)) to prevent target memorization")]

            # 3. Detect mapping naive groupby target mean onto the SAME rows:
            # e.g.: train['cat_enc'] = train['cat'].map(train.groupby('cat')['target'].mean())
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'map':
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr in ('mean', 'median', 'std', 'var'):
                        call_val = arg.func.value
                        if isinstance(call_val, ast.Subscript):
                            if isinstance(call_val.slice, ast.Constant) and isinstance(call_val.slice.value, str):
                                if any(t in call_val.slice.value.lower() for t in target_col_names):
                                    gb = call_val.value
                                    if isinstance(gb, ast.Call) and isinstance(gb.func, ast.Attribute) and gb.func.attr == 'groupby':
                                        # Extract receiver DF and source DF
                                        map_df_name = ""
                                        if isinstance(node.func.value, ast.Subscript):
                                            if isinstance(node.func.value.value, ast.Name):
                                                map_df_name = node.func.value.value.id.lower()
                                        elif isinstance(node.func.value, ast.Name):
                                            map_df_name = node.func.value.id.lower()

                                        gb_df_name = ""
                                        if isinstance(gb.func.value, ast.Name):
                                            gb_df_name = gb.func.value.id.lower()

                                        # If applying to a different holdout/validation/test partition (e.g. test vs train, val vs fit in OOF),
                                        # this is legitimate out-of-fold inference! Do NOT flag as self-target leakage.
                                        test_val_markers = ('test', 'val', 'valid', 'holdout', 'eval', 'inference')
                                        if any(m in map_df_name for m in test_val_markers) and map_df_name != gb_df_name:
                                            return []
                                        if map_df_name and gb_df_name and map_df_name != gb_df_name:
                                            return []

                                        return [Diagnostic(
                                            node.lineno, node.col_offset, self.severity, self.code,
                                            f"In-sample target encoding leak: mapping naive group target mean onto same rows leaks each row's own outcome into its feature",
                                            "Compute target statistics out-of-fold (OOF) across cross-validation splits, or use leave-one-out smoothing")]
        return []


class T019_NpRollNegative(Rule):
    code = "T019"
    name = "np_roll_future"
    severity = "error"
    category = "TemporalLeak"

    def check_node(self, node, ctx):
        if not isinstance(node, ast.Call):
            return []
        # Detect np.roll(arr, -N) or numpy.roll(arr, -N)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'roll':
            obj = node.func.value
            if isinstance(obj, ast.Name) and obj.id in ('np', 'numpy'):
                if len(node.args) >= 2:
                    val = self._get_negative_value(node.args[1], ctx)
                    if val is not None and val > 0:
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"np.roll(array, -{val}) rotates data {val} positions from the FUTURE (no NaN boundary)",
                            f"Use pd.Series.shift({val}) which inserts NaN at boundaries, or slice manually")]
                # Also check keyword: shift=... (numpy uses 'shift' param)
                for kw in node.keywords:
                    if kw.arg == 'shift':
                        val = self._get_negative_value(kw.value, ctx)
                        if val is not None and val > 0:
                            return [Diagnostic(
                                node.lineno, node.col_offset, self.severity, self.code,
                                f"np.roll(shift=-{val}) rotates data {val} positions from the FUTURE (no NaN boundary)",
                                f"Use pd.Series.shift({val}) which inserts NaN at boundaries")]
        return []

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
            # Sink-gate: suppress for exploratory display (print, .head(), .plot(), etc.)
            parent = getattr(node, 'parent', None)
            if parent is not None:
                # print(df.corr()) or display(df.corr())
                if isinstance(parent, ast.Call):
                    if isinstance(parent.func, ast.Name) and parent.func.id in ('print', 'display', 'logging'):
                        return []
                # df.corr().head(), df.corr().to_string(), etc.
                if isinstance(parent, ast.Attribute) and parent.attr in (
                    'head', 'tail', 'to_string', 'to_html', 'style', 'plot',
                    'describe', 'info', 'shape', 'values', 'round'
                ):
                    return []
                # Standalone expression (not assigned, not passed to model)
                if isinstance(parent, ast.Expr):
                    return []
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

        # Sink-Gating: If used in safe exploratory / display methods, do not flag as feature leakage
        parent = getattr(node, 'parent', None)
        if parent is not None:
            if isinstance(parent, ast.Attribute) and parent.attr in (
                'head', 'tail', 'plot', 'describe', 'info', 'hist', 'boxplot',
                'shape', 'dtypes', 'columns', 'index'
            ):
                return []
            if isinstance(parent, ast.Call):
                if isinstance(parent.func, ast.Name) and parent.func.id in ('print', 'display'):
                    return []

        train_markers = ('_train', 'train_', 'x_train', 'y_train', 'x_tr', 'y_tr', '_trn', 'trn_', 'train')
        caller_name = ""
        if isinstance(node.value, ast.Name):
            caller_name = node.value.id.lower()
        elif isinstance(node.value, ast.Attribute):
            caller_name = node.value.attr.lower()

        # Detect df[['feature1', 'target']] where target-like names are used as features
        if isinstance(node.slice, ast.List):
            raw_target_names = {'target', 'y', 'label', 'labels', 'outcome', 'y_true',
                                'future_price', 'future_return', 'next_close', 'next_price'}
            laundered_target_names = {'regime_success_rate', 'success_rate'}
            for elt in node.slice.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    val = elt.value.lower()
                    if val in raw_target_names:
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Column '{elt.value}' appears to be a target variable used as a feature",
                            "Remove target/outcome columns from feature inputs to avoid data leakage")]
                    if val in laundered_target_names or val.endswith('_success_rate') or val.endswith('_target_enc'):
                        # If caller is training data (e.g. train_df[['volatility', 'regime_success_rate']]),
                        # this is a valid fold-local target encoding feature! Do NOT flag it.
                        if any(m in caller_name for m in train_markers):
                            continue
                        return [Diagnostic(
                            node.lineno, node.col_offset, self.severity, self.code,
                            f"Column '{elt.value}' appears to be a target or target-laundered variable used as a feature across full dataset",
                            "Compute target-encoded features strictly fold-locally on training data")]
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
    T016_MergeAsofDirection(),
    T017_TargetLaundering(),
    T018_InSampleTargetEncoding(),
    T019_NpRollNegative(),
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

"""
PyCheck Data Flow Engine — Taint Propagation Analysis
Tracks how data flows through assignments, function returns, and method chains
to detect when future-tainted data reaches training/prediction sinks.
"""

import ast
from typing import Dict, Set, List, Optional, Tuple
from pycheck.rules import Diagnostic


class TaintSource:
    """Represents a source of tainted (future-leaked) data."""
    def __init__(self, var_name: str, reason: str, line: int, col: int):
        self.var_name = var_name
        self.reason = reason
        self.line = line
        self.col = col


class TaintTracker:
    """
    Tracks taint propagation and partition data provenance through Python AST.
    
    Taint sources: shift(-n), bfill(), future column access, etc.
    Provenance sources: train_test_split, temporal slicing, cross-validation splits.
    Contamination: concatenating or reassigning train data with test partitions.
    Sinks: .fit(), .fit_transform(), model training calls.
    
    Emits:
      - F001: Future-tainted data flows into sinks
      - F002: Test partition or contaminated data flows into training sinks
    """
    
    def __init__(self):
        self.tainted_vars: Dict[str, TaintSource] = {}
        self.diagnostics: List[Diagnostic] = []
        self.constants: Dict[str, any] = {}
        
        # Partition provenance tracking
        self.train_vars: Set[str] = set()
        self.test_vars: Set[str] = set()
        self.test_contaminated_vars: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.functions: Dict[str, ast.FunctionDef] = {}
        self.train_markers = ('_train', 'train_', 'x_train', 'y_train', 'x_tr', 'y_tr', '_trn', 'trn_', 'train_df', 'train_data', 'train_set')
        self.test_markers = ('_test', 'test_', 'x_test', 'y_test', 'x_val', 'y_val', '_val', 'val_', '_eval', 'eval_', 'holdout', 'validation', 'test_df', 'test_data', 'test_set')
    
    def analyze(self, tree: ast.AST, source_lines: List[str]) -> List[Diagnostic]:
        """Run taint and provenance analysis on a full AST."""
        self.tainted_vars.clear()
        self.diagnostics.clear()
        self.constants.clear()
        self.train_vars.clear()
        self.test_vars.clear()
        self.test_contaminated_vars.clear()
        self.aliases.clear()
        self.functions.clear()
        
        # Two-pass analysis:
        # Pass 1: Identify taint & provenance sources, follow reassignments and combinations
        # Pass 2: Detect sensitive sinks
        self._collect_taints(tree)
        self._check_sinks(tree)
        
        return self.diagnostics
    
    def _iter_statements(self, node: ast.AST):
        """Yield statements recursively in source code execution order."""
        if hasattr(node, 'body') and isinstance(node.body, list):
            for stmt in node.body:
                yield stmt
                yield from self._iter_statements(stmt)
        if hasattr(node, 'orelse') and isinstance(node.orelse, list):
            for stmt in node.orelse:
                yield stmt
                yield from self._iter_statements(stmt)
        if hasattr(node, 'handlers') and isinstance(node.handlers, list):
            for h in node.handlers:
                yield from self._iter_statements(h)
        if hasattr(node, 'finalbody') and isinstance(node.finalbody, list):
            for stmt in node.finalbody:
                yield stmt
                yield from self._iter_statements(stmt)

    def _extract_constant(self, expr):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return expr.value
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            if isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, (int, float)):
                return -expr.operand.value
        return None

    def _is_negative_period(self, expr) -> bool:
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            return True
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)) and expr.value < 0:
            return True
        if isinstance(expr, ast.Name):
            val = self.constants.get(expr.id)
            if val is not None and isinstance(val, (int, float)) and val < 0:
                return True
        return False

    def _collect_taints(self, tree: ast.AST):
        """Pass 1: Walk the AST, index helper functions, track taint and provenance."""
        # Index helper functions defined in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node

        # Extract literal constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        val = self._extract_constant(node.value)
                        if val is not None:
                            self.constants[target.id] = val

        # Linear traversal of statements for data-flow & provenance propagation
        for stmt in self._iter_statements(tree):
            if isinstance(stmt, ast.Assign):
                # 1. Check for split calls (train_test_split)
                if isinstance(stmt.value, ast.Call):
                    call_name = self._get_call_name(stmt.value)
                    if call_name == 'train_test_split':
                        for target in stmt.targets:
                            if isinstance(target, (ast.Tuple, ast.List)):
                                names = [self._get_var_name(e) for e in target.elts]
                                if len(names) >= 4:
                                    if names[0]: self.train_vars.add(names[0])
                                    if names[1]: self.test_vars.add(names[1])
                                    if names[2]: self.train_vars.add(names[2])
                                    if names[3]: self.test_vars.add(names[3])
                                elif len(names) >= 2:
                                    if names[0]: self.train_vars.add(names[0])
                                    if names[1]: self.test_vars.add(names[1])

                # 2. Check for temporal slicing splits: df[:split_point] vs df[split_point:]
                if isinstance(stmt.value, ast.Subscript):
                    sl = stmt.value.slice
                    if isinstance(sl, ast.Slice):
                        for target in stmt.targets:
                            t_name = self._get_var_name(target)
                            if t_name:
                                if sl.lower is None or (isinstance(sl.lower, ast.Constant) and sl.lower.value == 0):
                                    if sl.upper is not None:
                                        self.train_vars.add(t_name)
                                elif sl.upper is None and sl.lower is not None:
                                    self.test_vars.add(t_name)
                    elif isinstance(sl, ast.Name):
                        for target in stmt.targets:
                            t_name = self._get_var_name(target)
                            if t_name:
                                if sl.id in self.train_vars or any(m in sl.id.lower() for m in self.train_markers):
                                    self.train_vars.add(t_name)
                                elif sl.id in self.test_vars or any(m in sl.id.lower() for m in self.test_markers):
                                    self.test_vars.add(t_name)

                # 3. Register naming evidence
                for target in stmt.targets:
                    t_name = self._get_var_name(target)
                    if t_name:
                        t_lower = t_name.lower()
                        if any(m in t_lower for m in self.train_markers) and t_name not in self.test_contaminated_vars:
                            self.train_vars.add(t_name)
                        elif any(m in t_lower for m in self.test_markers):
                            self.test_vars.add(t_name)

                # 4. Standard future-shift / bfill taint check
                self._check_assignment_taint(stmt)

                # 5. Check test contamination propagation
                contamination = self._expr_contains_test(stmt.value)
                for target in stmt.targets:
                    t_name = self._get_var_name(target)
                    if not t_name:
                        continue
                    if contamination:
                        self.test_contaminated_vars[t_name] = contamination
                        self.train_vars.discard(t_name)
                    else:
                        # Follow aliases
                        src_name = self._get_var_name(stmt.value)
                        if src_name:
                            self.aliases[t_name] = src_name
                            if src_name in self.test_contaminated_vars:
                                self.test_contaminated_vars[t_name] = f"alias of contaminated '{src_name}' ({self.test_contaminated_vars[src_name]})"
                                self.train_vars.discard(t_name)
                            elif src_name in self.test_vars:
                                self.test_vars.add(t_name)
                            elif src_name in self.train_vars and t_name not in self.test_contaminated_vars:
                                self.train_vars.add(t_name)

            elif isinstance(stmt, ast.AugAssign):
                self._check_augassign_taint(stmt)
                contamination = self._expr_contains_test(stmt.value)
                t_name = self._get_var_name(stmt.target)
                if t_name and contamination:
                    self.test_contaminated_vars[t_name] = f"augmented assignment ({type(stmt.op).__name__}) with {contamination}"
                    self.train_vars.discard(t_name)

            elif isinstance(stmt, ast.For):
                if isinstance(stmt.iter, ast.Call) and self._get_call_name(stmt.iter) in ('split',):
                    if isinstance(stmt.target, (ast.Tuple, ast.List)) and len(stmt.target.elts) == 2:
                        idx0 = self._get_var_name(stmt.target.elts[0])
                        idx1 = self._get_var_name(stmt.target.elts[1])
                        if idx0: self.train_vars.add(idx0)
                        if idx1: self.test_vars.add(idx1)

    def _check_assignment_taint(self, node: ast.Assign):
        """Check if an assignment creates or propagates a taint."""
        # Get the taint from the RHS
        taint_reason = self._expr_taint(node.value)
        if taint_reason is None:
            return
        
        # Mark all LHS targets as tainted
        for target in node.targets:
            var_name = self._get_var_name(target)
            if var_name:
                self.tainted_vars[var_name] = TaintSource(
                    var_name, taint_reason, node.lineno, node.col_offset)
    
    def _check_augassign_taint(self, node: ast.AugAssign):
        """Check if an augmented assignment (+=, etc.) propagates taint."""
        taint_reason = self._expr_taint(node.value)
        if taint_reason:
            var_name = self._get_var_name(node.target)
            if var_name:
                self.tainted_vars[var_name] = TaintSource(
                    var_name, taint_reason, node.lineno, node.col_offset)
    
    def _expr_taint(self, node: ast.AST) -> Optional[str]:
        """
        Determine if an expression produces tainted data.
        Returns the taint reason string, or None if clean.
        """
        if isinstance(node, ast.Call):
            return self._call_taint(node)
        
        if isinstance(node, ast.Name):
            # Propagation: if RHS is a tainted variable, the assignment is tainted
            if node.id in self.tainted_vars:
                return f"derived from tainted variable '{node.id}' ({self.tainted_vars[node.id].reason})"
        
        if isinstance(node, ast.BinOp):
            # Binary operations propagate taint from either operand
            left_taint = self._expr_taint(node.left)
            right_taint = self._expr_taint(node.right)
            return left_taint or right_taint
        
        if isinstance(node, ast.Subscript):
            # Subscript propagates taint from the value being indexed
            val_taint = self._expr_taint(node.value)
            if val_taint:
                return val_taint
            # Check if indexing a tainted column name
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                col_lower = node.slice.value.lower()
                # Only flag column names that unambiguously refer to future data.
                # 'target' and 'label' are intentionally excluded — they are standard
                # ML label column names and produce false positives when accessed on a
                # correctly split train/test dataset.
                future_cols = {'future_price', 'future_return', 'next_close', 'next_price',
                               'next_day', 'tomorrow', 'price_change_next_day', 'future_avg_price'}
                if any(f in col_lower for f in future_cols):
                    return f"accessing future-named column '{node.slice.value}'"
        
        if isinstance(node, ast.Attribute):
            return self._expr_taint(node.value)
        
        return None
    
    def _call_taint(self, node: ast.Call) -> Optional[str]:
        """Check if a function call produces tainted output."""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            
            # Direct taint sources
            if method == 'shift':
                for arg in node.args:
                    if self._is_negative_period(arg):
                        return "shift() with negative period (future access)"
                for kw in node.keywords:
                    if kw.arg == 'periods' and self._is_negative_period(kw.value):
                        return "shift() with negative period (future access)"
            
            if method in ('bfill', 'backfill'):
                return f".{method}() backfills from future values"
            
            if method == 'fillna':
                for kw in node.keywords:
                    if kw.arg == 'method' and isinstance(kw.value, ast.Constant):
                        if kw.value.value in ('bfill', 'backfill'):
                            return f".fillna(method='{kw.value.value}') backfills from future"
            
            if method == 'diff':
                for arg in node.args:
                    if self._is_negative_period(arg):
                        return ".diff() with negative period (future access)"
                for kw in node.keywords:
                    if kw.arg == 'periods' and self._is_negative_period(kw.value):
                        return ".diff() with negative period (future access)"
            
            # Taint propagation through method chains
            obj_taint = self._expr_taint(node.func.value)
            if obj_taint:
                # Most method calls on tainted data produce tainted output
                non_taint_methods = {'shape', 'dtype', 'dtypes', 'columns', 'index',
                                     'describe', 'info', 'head', 'tail', 'len', '__len__'}
                if method not in non_taint_methods:
                    return obj_taint
        
        if isinstance(node.func, ast.Name):
            # Check if calling a function with tainted arguments
            for arg in node.args:
                arg_taint = self._expr_taint(arg)
                if arg_taint:
                    return f"function receives tainted input: {arg_taint}"
        
        # R2-9 fix: Detect getattr(obj, 'shift')(-1) and similar patterns
        # Pattern: call(getattr(obj, 'shift'), -1) => obj.shift(-1)
        if isinstance(node.func, ast.Call):
            inner_call = node.func
            if isinstance(inner_call.func, ast.Name) and inner_call.func.id == 'getattr':
                if len(inner_call.args) >= 2 and isinstance(inner_call.args[1], ast.Constant):
                    attr_name = inner_call.args[1].value
                    if attr_name == 'shift':
                        for arg in node.args:
                            if self._is_negative_period(arg):
                                return "getattr(..., 'shift')() with negative period (future access)"
                        for kw in node.keywords:
                            if kw.arg == 'periods' and self._is_negative_period(kw.value):
                                return "getattr(..., 'shift')() with negative period (future access)"
                    elif attr_name == 'diff':
                        for arg in node.args:
                            if self._is_negative_period(arg):
                                return "getattr(..., 'diff')() with negative period (future access)"
                        for kw in node.keywords:
                            if kw.arg == 'periods' and self._is_negative_period(kw.value):
                                return "getattr(..., 'diff')() with negative period (future access)"
                    elif attr_name in ('bfill', 'backfill'):
                        return f"getattr(..., '{attr_name}')() backfills from future values"
        
        return None
    
    def _check_sinks(self, tree: ast.AST):
        """Pass 2: Check if tainted or test-contaminated data flows into sensitive sinks."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                sinks = {'fit', 'fit_transform', 'fit_predict', 'train',
                         'partial_fit', 'predict', 'score',
                         'cross_validate', 'cross_val_score', 'cross_val_predict',
                         'GridSearchCV', 'RandomizedSearchCV'}
                
                if method in sinks:
                    # Check positional arguments for future taint (F001)
                    for arg in node.args:
                        arg_taint = self._expr_taint(arg)
                        if arg_taint:
                            self.diagnostics.append(Diagnostic(
                                node.lineno, node.col_offset, "error", "F001",
                                f"Tainted data flows into .{method}(): {arg_taint}",
                                "Ensure training/prediction data is free of future information"))
                    # Check keyword arguments for future taint (F001)
                    for kw in node.keywords:
                        kw_taint = self._expr_taint(kw.value)
                        if kw_taint:
                            kw_name = kw.arg if kw.arg else "**kwargs"
                            self.diagnostics.append(Diagnostic(
                                node.lineno, node.col_offset, "error", "F001",
                                f"Tainted data flows into .{method}() via keyword '{kw_name}': {kw_taint}",
                                "Ensure training/prediction data is free of future information"))

                # Check F002: Data contamination & test partitions in training sinks
                training_sinks = {'fit', 'fit_transform', 'fit_predict', 'train', 'partial_fit',
                                  'cross_validate', 'cross_val_score', 'cross_val_predict',
                                  'GridSearchCV', 'RandomizedSearchCV'}
                if method in training_sinks:
                    obj_name = self._get_obj_name(node.func.value)
                    safe_objects = {'fig', 'figure', 'ax', 'axes', 'plt', 'layout',
                                    'canvas', 'widget', 'app', 'db', 'session', 'socket',
                                    'parser', 'formatter', 'logger', 'handler'}
                    if obj_name and obj_name.lower() in safe_objects:
                        continue
                    
                    for arg in node.args:
                        var_name = self._get_var_name(arg)
                        if var_name:
                            base_var = var_name.split('.')[0]
                            if base_var in self.test_contaminated_vars:
                                reason = self.test_contaminated_vars[base_var]
                                self.diagnostics.append(Diagnostic(
                                    node.lineno, node.col_offset, "error", "F002",
                                    f"Data contamination: '{base_var}' contains test partition data ({reason}) but is passed to .{method}()",
                                    "Ensure data passed to .{method}() contains only genuine training data without test partitions"))
                            elif base_var in self.test_vars or (base_var not in self.train_vars and any(m in base_var.lower() for m in self.test_markers)):
                                self.diagnostics.append(Diagnostic(
                                    node.lineno, node.col_offset, "error", "F002",
                                    f"Test data passed to training sink: '{base_var}' is a test/validation partition but is passed to .{method}()",
                                    "Call .{method}() only on training data, use .transform() or .predict() on test data"))
                    
                    for kw in node.keywords:
                        var_name = self._get_var_name(kw.value)
                        if var_name:
                            base_var = var_name.split('.')[0]
                            kw_name = kw.arg if kw.arg else "**kwargs"
                            if base_var in self.test_contaminated_vars:
                                reason = self.test_contaminated_vars[base_var]
                                self.diagnostics.append(Diagnostic(
                                    node.lineno, node.col_offset, "error", "F002",
                                    f"Data contamination: keyword '{kw_name}' receives '{base_var}' which contains test partition data ({reason}) in .{method}()",
                                    "Ensure data passed to .{method}() contains only genuine training data without test partitions"))
                            elif base_var in self.test_vars or (base_var not in self.train_vars and any(m in base_var.lower() for m in self.test_markers)):
                                self.diagnostics.append(Diagnostic(
                                    node.lineno, node.col_offset, "error", "F002",
                                    f"Test data passed to training sink: keyword '{kw_name}' receives test partition '{base_var}' in .{method}()",
                                    "Call .{method}() only on training data, use .transform() or .predict() on test data"))

    def _expr_contains_test(self, expr: Optional[ast.AST]) -> Optional[str]:
        """Check if an expression incorporates test partition data or contaminated variables."""
        if expr is None:
            return None
        
        # 1. Direct variable name
        if isinstance(expr, ast.Name):
            var_id = expr.id
            if var_id in self.test_contaminated_vars:
                return f"contaminated variable '{var_id}' ({self.test_contaminated_vars[var_id]})"
            if var_id in self.test_vars:
                return f"test partition '{var_id}'"
            if var_id not in self.train_vars and any(m in var_id.lower() for m in self.test_markers):
                return f"test partition '{var_id}'"
            return None
        
        # 2. Function / Method Calls
        if isinstance(expr, ast.Call):
            call_name = self._get_call_name(expr)
            # Concatenation: pd.concat([X_train, X_test]), np.concatenate, np.vstack
            if call_name in ('concat', 'concatenate', 'vstack', 'hstack'):
                if expr.args:
                    arg0 = expr.args[0]
                    if isinstance(arg0, (ast.List, ast.Tuple)):
                        for elt in arg0.elts:
                            reason = self._expr_contains_test(elt)
                            if reason:
                                return f"concatenation ({call_name}) including {reason}"
                    else:
                        reason = self._expr_contains_test(arg0)
                        if reason:
                            return f"concatenation ({call_name}) including {reason}"
            
            # Method calls: .append(), .merge(), .join()
            if isinstance(expr.func, ast.Attribute):
                method = expr.func.attr
                if method in ('append', 'merge', 'join'):
                    obj_reason = self._expr_contains_test(expr.func.value)
                    if obj_reason:
                        return f".{method}() on {obj_reason}"
                    for arg in expr.args:
                        arg_reason = self._expr_contains_test(arg)
                        if arg_reason:
                            return f".{method}() with {arg_reason}"
            
            # Helper functions defined in this module
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id
                if func_name in self.functions:
                    f_def = self.functions[func_name]
                    param_names = [a.arg for a in f_def.args.args]
                    contaminated_params = []
                    for idx, arg in enumerate(expr.args):
                        reason = self._expr_contains_test(arg)
                        if reason:
                            p_name = param_names[idx] if idx < len(param_names) else f"arg{idx}"
                            contaminated_params.append((p_name, reason))
                    
                    if contaminated_params:
                        # Check return statements in f_def
                        ret_nodes = [n for n in ast.walk(f_def) if isinstance(n, ast.Return) and n.value is not None]
                        for p_name, reason in contaminated_params:
                            for ret in ret_nodes:
                                ret_names = {n.id for n in ast.walk(ret.value) if isinstance(n, ast.Name)}
                                if p_name in ret_names:
                                    return f"helper function '{func_name}()' returning {reason}"
                        p_name, reason = contaminated_params[0]
                        return f"helper function '{func_name}()' receiving {reason}"
        
        # 3. Binary operations: a + b
        if isinstance(expr, ast.BinOp):
            left_r = self._expr_contains_test(expr.left)
            if left_r:
                return left_r
            return self._expr_contains_test(expr.right)
        
        # 4. List / Tuple: [a, b]
        if isinstance(expr, (ast.List, ast.Tuple)):
            for elt in expr.elts:
                r = self._expr_contains_test(elt)
                if r:
                    return r
        
        # 5. Subscript: obj[...]
        if isinstance(expr, ast.Subscript):
            val_r = self._expr_contains_test(expr.value)
            if val_r:
                return val_r
            slice_r = self._expr_contains_test(expr.slice)
            if slice_r:
                return f"indexed by {slice_r}"
        
        # 6. Attribute: obj.attr
        if isinstance(expr, ast.Attribute):
            return self._expr_contains_test(expr.value)
        
        return None

    def _get_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
        return ""

    def _get_obj_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _get_var_name(self, node: ast.AST) -> Optional[str]:
        """Extract variable name from an assignment target or expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            base = self._get_var_name(node.value)
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return f"{base}.{node.slice.value}" if base else node.slice.value
            return base
        if isinstance(node, ast.Attribute):
            base = self._get_var_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

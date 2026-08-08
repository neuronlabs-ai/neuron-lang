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
    Tracks taint propagation through Python AST.
    
    Taint sources: shift(-n), bfill(), future column access, etc.
    Taint sinks: .fit(), .predict(), model training calls.
    
    When tainted data flows into a sink, a diagnostic is emitted.
    """
    
    def __init__(self):
        self.tainted_vars: Dict[str, TaintSource] = {}
        self.diagnostics: List[Diagnostic] = []
    
    def analyze(self, tree: ast.AST, source_lines: List[str]) -> List[Diagnostic]:
        """Run taint analysis on a full AST."""
        self.tainted_vars.clear()
        self.diagnostics.clear()
        
        # Two-pass analysis:
        # Pass 1: Identify all taint sources (assignments from leaky operations)
        # Pass 2: Track propagation and detect sinks
        self._collect_taints(tree)
        self._check_sinks(tree)
        
        return self.diagnostics
    
    def _collect_taints(self, tree: ast.AST):
        """Pass 1: Walk the AST and mark variables that receive tainted data."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                self._check_assignment_taint(node)
            elif isinstance(node, ast.AugAssign):
                self._check_augassign_taint(node)
    
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
                future_cols = {'future_price', 'future_return', 'next_close', 'next_price',
                               'target', 'next_day', 'tomorrow', 'future'}
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
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        return "shift() with negative period (future access)"
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        if arg.value < 0:
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
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        return ".diff() with negative period (future access)"
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
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
        
        return None
    
    def _check_sinks(self, tree: ast.AST):
        """Pass 2: Check if tainted data flows into sensitive sinks."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                sinks = {'fit', 'fit_transform', 'fit_predict', 'train',
                          'partial_fit', 'predict', 'score'}
                
                if method in sinks:
                    # Check if any argument is tainted
                    for arg in node.args:
                        arg_taint = self._expr_taint(arg)
                        if arg_taint:
                            self.diagnostics.append(Diagnostic(
                                node.lineno, node.col_offset, "error", "F001",
                                f"Tainted data flows into .{method}(): {arg_taint}",
                                "Ensure training/prediction data is free of future information"))
    
    def _get_var_name(self, node: ast.AST) -> Optional[str]:
        """Extract variable name from an assignment target."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            # df['col_name'] — return as df.col_name
            base = self._get_var_name(node.value)
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return f"{base}.{node.slice.value}" if base else node.slice.value
            return base
        if isinstance(node, ast.Attribute):
            base = self._get_var_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

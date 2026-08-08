"""
PyCheck Analyzer — ML Safety Static Analysis Engine
Uses a rule registry + taint flow analysis to detect temporal leaks,
causal confusion, and uncertainty bugs in Python ML scripts.
"""

import ast
import sys
import json
from typing import List, Tuple

from pycheck.rules import ALL_RULES, RULES_BY_CODE, AnalysisContext, Diagnostic
from pycheck.flow import TaintTracker


class PyCheckAnalyzer(ast.NodeVisitor):
    """
    AST visitor that runs all registered rules against each node.
    Also tracks loop context for loop-aware rules.
    """
    
    def __init__(self, source_lines: List[str], filepath: str):
        self.ctx = AnalysisContext(source_lines, filepath)
        self.diagnostics: List[Diagnostic] = []
        self.rules = ALL_RULES
    
    def _run_rules(self, node: ast.AST):
        """Run all rules against a single AST node."""
        for rule in self.rules:
            results = rule.check_node(node, self.ctx)
            self.diagnostics.extend(results)
    
    def generic_visit(self, node):
        """Override to run rules on every node."""
        self._run_rules(node)
        super().generic_visit(node)
    
    def visit_For(self, node):
        """Track loop context for rules that need it."""
        old_in_loop = self.ctx.in_loop
        old_loop_var = self.ctx.loop_var
        
        self.ctx.in_loop = True
        if isinstance(node.target, ast.Name):
            self.ctx.loop_var = node.target.id
        
        self._run_rules(node)
        # Visit body with loop context
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        
        self.ctx.in_loop = old_in_loop
        self.ctx.loop_var = old_loop_var
    
    def visit_While(self, node):
        """Track loop context."""
        old_in_loop = self.ctx.in_loop
        self.ctx.in_loop = True
        
        self._run_rules(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        
        self.ctx.in_loop = old_in_loop


def analyze_file(filepath: str) -> Tuple[List[dict], List[str]]:
    """
    Analyze a Python file for ML safety issues.
    Returns (diagnostics_list, source_lines).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    source_lines = source.split('\n')
    tree = ast.parse(source, filename=filepath)
    
    # Phase 1: Rule-based AST analysis
    analyzer = PyCheckAnalyzer(source_lines, filepath)
    analyzer.visit(tree)
    
    # Phase 2: Data flow taint analysis
    tracker = TaintTracker()
    flow_diagnostics = tracker.analyze(tree, source_lines)
    
    # Combine and deduplicate
    all_diagnostics = analyzer.diagnostics + flow_diagnostics
    
    # Deduplicate by (line, code)
    seen = set()
    unique = []
    for d in all_diagnostics:
        key = (d.line, d.code)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    
    # Sort by line number
    unique.sort(key=lambda d: (d.line, d.col))
    
    return [d.to_dict() for d in unique], source_lines


def format_output(filepath: str, diagnostics: List[dict], source_lines: List[str],
                   show_info: bool = False):
    """Format diagnostics as human-readable output."""
    
    # Filter by severity
    if not show_info:
        diagnostics = [d for d in diagnostics if d['severity'] != 'info']
    
    errors = [d for d in diagnostics if d['severity'] == 'error']
    warnings = [d for d in diagnostics if d['severity'] == 'warning']
    infos = [d for d in diagnostics if d['severity'] == 'info']
    
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', errors='replace').decode('ascii'))
    
    header_sep = "=" * 65
    line_sep = "-" * 65
    
    safe_print(f"\n{header_sep}")
    safe_print(f"  PyCheck — NEURON ML Safety Analyzer")
    safe_print(f"  Scanning: {filepath}")
    safe_print(f"  Rules: {len(ALL_RULES)} active")
    safe_print(f"{header_sep}\n")
    
    if not diagnostics:
        safe_print("  [OK] No issues found.\n")
        return
    
    safe_print(f"  {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info(s)\n")
    
    for d in diagnostics:
        line_num = d['line']
        
        # Color-like severity prefix
        severity_label = d['severity'].upper()
        safe_print(f"  {severity_label}[{d['code']}]: {d['message']}")
        safe_print(f"  --> {filepath}:{line_num}:{d['col']}")
        
        if 0 < line_num <= len(source_lines):
            line_text = source_lines[line_num - 1]
            safe_print(f"   {line_num:>3} |  {line_text}")
            pointer_start = d['col']
            safe_print(f"       {' ' * pointer_start}^^^^^^^^^^^")
        
        if 'help' in d:
            safe_print(f"       help: {d['help']}")
        
        safe_print("")
    
    safe_print(f"{line_sep}")
    safe_print(f"  Summary: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info(s)")
    if errors:
        safe_print(f"  These {len(errors)} error(s) would be COMPILE-TIME ERRORS in NEURON")
    safe_print(f"  Python detected: 0 of these issues at runtime")
    safe_print(f"{line_sep}\n")

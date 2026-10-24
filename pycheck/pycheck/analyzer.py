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

    def visit_Assign(self, node):
        """Track literal constant assignments for constant folding / alias analysis."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                val = None
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                    val = node.value.value
                elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                    if isinstance(node.value.operand, ast.Constant) and isinstance(node.value.operand.value, (int, float)):
                        val = -node.value.operand.value
                if val is not None:
                    self.ctx.constants[target.id] = val
                self.ctx.assignments[target.id] = node.value
        self._run_rules(node)
        self.generic_visit(node)
    
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



def parse_suppressions(source_lines: List[str]) -> dict:
    """
    Parse inline and line-above suppression comments:
    # pycheck: ignore[T001, C002] reason: ...
    # pycheck: ignore
    # noqa: T001
    """
    import re
    suppressions = {}
    pattern = re.compile(
        r'#\s*(?:pycheck:\s*ignore(?:\[([a-zA-Z0-9,\s]+)\])?|noqa(?::\s*([a-zA-Z0-9,\s]+))?)',
        re.IGNORECASE
    )
    for lineno, line in enumerate(source_lines, start=1):
        match = pattern.search(line)
        if match:
            rules_str = match.group(1) or match.group(2)
            codes = {c.strip().upper() for c in rules_str.split(',') if c.strip()} if rules_str else {'ALL'}
            suppressions.setdefault(lineno, set()).update(codes)
            # If the comment is on its own standalone line, also apply to the next code line
            if line.strip().startswith('#') and lineno + 1 <= len(source_lines):
                suppressions.setdefault(lineno + 1, set()).update(codes)
    return suppressions

def analyze_code(source: str, filepath: str = "<string>") -> Tuple[List[dict], List[str]]:
    """
    Analyze Python source code for ML safety issues.
    Returns (diagnostics_list, source_lines).
    """
    source_lines = source.split('\n')
    tree = ast.parse(source, filename=filepath)
    # Link parent references across the AST for sink-gated rule analysis
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent
    
    # Phase 1: Data flow taint & provenance analysis
    tracker = TaintTracker()
    flow_diagnostics = tracker.analyze(tree, source_lines)
    
    # Phase 2: Rule-based AST analysis (with provenance populated in ctx)
    analyzer = PyCheckAnalyzer(source_lines, filepath)
    analyzer.ctx.tree = tree
    analyzer.ctx.tainted_vars = {k: v.reason for k, v in tracker.tainted_vars.items()}
    analyzer.ctx.train_vars = tracker.train_vars
    analyzer.ctx.test_vars = tracker.test_vars
    analyzer.ctx.test_contaminated_vars = tracker.test_contaminated_vars
    analyzer.visit(tree)
    
    # Combine and deduplicate
    all_diagnostics = analyzer.diagnostics + flow_diagnostics
    
    suppressed_map = parse_suppressions(source_lines)

    # Deduplicate and filter suppressed diagnostics
    seen = set()
    unique = []
    for d in all_diagnostics:
        key = (d.line, d.code)
        if key not in seen:
            seen.add(key)
            line_suppressed = suppressed_map.get(d.line, set())
            if 'ALL' not in line_suppressed and d.code not in line_suppressed:
                unique.append(d)
    
    # Sort by line number
    unique.sort(key=lambda d: (d.line, d.col))
    
    return [d.to_dict() for d in unique], source_lines


def analyze_file(filepath: str) -> Tuple[List[dict], List[str]]:
    """
    Analyze a Python (.py) or Jupyter notebook (.ipynb) file for ML safety issues.
    Returns (diagnostics_list, source_lines).
    """
    if filepath.endswith('.ipynb'):
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_lines = []
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                raw = cell.get('source', [])
                if isinstance(raw, str):
                    raw = raw.splitlines(True)
                for line in raw:
                    s_line = line.strip()
                    if s_line.startswith('%') or s_line.startswith('!'):
                        code_lines.append(f"# {line}")
                    else:
                        code_lines.append(line)
                code_lines.append('\n')
        source = "".join(code_lines)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
    
    return analyze_code(source, filepath=filepath)


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
    elif warnings:
        safe_print(f"  These {len(warnings)} temporal issue(s) are guarded at compile-time in NEURON")
    safe_print(f"  Python detected: 0 of these issues at runtime.")
    safe_print(f"  --> NEURON Zero-Leakage Language: https://github.com/neuronlabs-ai/neuron-lang")
    safe_print(f"  --> PyCheck Documentation & CI/CD: https://github.com/neuronlabs-ai/pycheck")
    safe_print(f"{line_sep}\n")

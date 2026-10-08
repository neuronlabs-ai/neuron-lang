"""
CLI Entrypoint for PyCheck
"""

import sys
import json
import os
from pycheck.analyzer import analyze_file, format_output
from pycheck.rules import ALL_RULES


def main():
    args = sys.argv[1:]
    
    if not args or '--help' in args or '-h' in args:
        print("PyCheck — NEURON ML Safety Analyzer")
        print(f"  {len(ALL_RULES)} rules for temporal leaks, causal confusion, and uncertainty bugs\n")
        print("Usage: pycheck <script.py> [options]\n")
        print("Options:")
        print("  --json       Output diagnostics as JSON")
        print("  --info       Include info-level diagnostics")
        print("  --quiet      Only show errors (no warnings)")
        print("  --list       List all available rules")
        print("  -h, --help   Show this help message")
        sys.exit(0)
    
    if '--list' in args:
        print(f"\nPyCheck Rules ({len(ALL_RULES)} total):\n")
        current_cat = ""
        for rule in ALL_RULES:
            if rule.category != current_cat:
                current_cat = rule.category
                print(f"\n  [{current_cat}]")
            print(f"    {rule.code}  {rule.severity:<8}  {rule.name}")
        print()
        sys.exit(0)
    
    # Get target path (first non-flag argument, defaults to '.' if flag only or omitted)
    filepath = None
    for a in args:
        if not a.startswith('-'):
            filepath = a
            break
    
    if not filepath:
        filepath = "."
    
    if not os.path.exists(filepath):
        print(f"Error: Path not found: {filepath}")
        sys.exit(1)
    
    show_info = '--info' in args
    is_quiet = '--quiet' in args
    is_json = '--json' in args

    # Collect target files
    files_to_scan = []
    if os.path.isdir(filepath):
        ignored_dirs = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'env', 'node_modules', 'dist', 'build', '.egg-info'}
        for root, dirs, files in os.walk(filepath):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
            for f in files:
                if f.endswith('.py') or f.endswith('.ipynb'):
                    files_to_scan.append(os.path.join(root, f))
    else:
        files_to_scan = [filepath]

    if not files_to_scan:
        print(f"No Python (.py) or notebook (.ipynb) files found in {filepath}")
        sys.exit(0)

    total_errors = 0
    total_warnings = 0
    files_with_issues = 0
    all_json_results = {}

    for f_path in files_to_scan:
        try:
            diagnostics, source_lines = analyze_file(f_path)
        except SyntaxError as e:
            if not is_json:
                print(f"Syntax error in {f_path}: {e}")
            total_errors += 1
            continue
        except Exception as e:
            if not is_json:
                print(f"Error analyzing {f_path}: {e}")
            continue

        if is_quiet:
            diagnostics = [d for d in diagnostics if d['severity'] == 'error']
        elif not show_info:
            diagnostics = [d for d in diagnostics if d['severity'] != 'info']

        err_count = sum(1 for d in diagnostics if d['severity'] == 'error')
        warn_count = sum(1 for d in diagnostics if d['severity'] == 'warning')
        total_errors += err_count
        total_warnings += warn_count

        if diagnostics:
            files_with_issues += 1
            if is_json:
                all_json_results[f_path] = diagnostics
            else:
                format_output(f_path, diagnostics, source_lines, show_info=show_info)

    if is_json:
        print(json.dumps(all_json_results, indent=2))
    elif len(files_to_scan) > 1:
        clean_files = len(files_to_scan) - files_with_issues
        print("=================================================================")
        print("  PyCheck Directory Summary")
        print(f"  Scanned: {len(files_to_scan)} files | Clean: {clean_files} | Flagged: {files_with_issues}")
        print(f"  Total: {total_errors} error(s), {total_warnings} warning(s)")
        print("=================================================================\n")

    if total_errors > 0:
        sys.exit(1)



if __name__ == '__main__':
    main()

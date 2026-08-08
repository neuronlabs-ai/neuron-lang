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
    
    # Get filepath (first non-flag argument)
    filepath = None
    for a in args:
        if not a.startswith('-'):
            filepath = a
            break
    
    if not filepath:
        print("Error: No file specified")
        sys.exit(1)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    try:
        diagnostics, source_lines = analyze_file(filepath)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        sys.exit(1)
    
    if '--json' in args:
        # Filter by severity
        if '--quiet' in args:
            diagnostics = [d for d in diagnostics if d['severity'] == 'error']
        elif '--info' not in args:
            diagnostics = [d for d in diagnostics if d['severity'] != 'info']
        print(json.dumps(diagnostics, indent=2))
    else:
        show_info = '--info' in args
        if '--quiet' in args:
            diagnostics = [d for d in diagnostics if d['severity'] == 'error']
        format_output(filepath, diagnostics, source_lines, show_info=show_info)
    
    # Exit with code 1 if any errors were found
    errors = [d for d in diagnostics if d['severity'] == 'error']
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()

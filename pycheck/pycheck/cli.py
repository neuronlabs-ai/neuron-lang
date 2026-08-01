"""
CLI Entrypoint for PyCheck
"""

import sys
import json
from pycheck.analyzer import analyze_file, format_output

def main():
    if len(sys.argv) < 2:
        print("Usage: pycheck <script.py> [--json]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    try:
        diagnostics, source_lines = analyze_file(filepath)
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        sys.exit(1)
        
    format_output(filepath, diagnostics, source_lines)
    
    if '--json' in sys.argv:
        print(json.dumps(diagnostics, indent=2))
    
    # Exit with code 1 if any errors were found
    errors = [d for d in diagnostics if d['severity'] == 'error']
    if errors:
        sys.exit(1)

if __name__ == '__main__':
    main()

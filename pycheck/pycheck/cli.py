import sys
import json
import os
import hashlib
from pycheck.analyzer import analyze_file, format_output
from pycheck.rules import ALL_RULES
from pycheck.audit import generate_proof_of_causality_certificate, format_certificate_text


def main():
    args = sys.argv[1:]
    
    if not args or '--help' in args or '-h' in args:
        print("PyCheck — NEURON ML Safety Analyzer")
        print(f"  {len(ALL_RULES)} rules for temporal leaks, causal confusion, and uncertainty bugs\n")
        print("Usage: pycheck <path> [options]\n")
        print("Options:")
        print("  --json               Output diagnostics as JSON")
        print("  --gitlab             Output diagnostics in GitLab Code Quality JSON format")
        print("  --audit, -a          Generate a verifiable Proof-of-Causality Audit Certificate")
        print("  -o, --output <file>  Write JSON / GitLab report directly to specified file")
        print("  --info               Include info-level diagnostics")
        print("  --quiet              Only show errors (no warnings)")
        print("  --list               List all available rules")
        print("  -h, --help           Show this help message")
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
    
    show_info = '--info' in args
    is_quiet = '--quiet' in args
    is_json = '--json' in args
    is_gitlab = '--gitlab' in args
    is_audit = '--audit' in args or '-a' in args or (len(args) > 0 and args[0] == 'audit')

    output_file = None
    if '--output' in args:
        idx = args.index('--output')
        if idx + 1 < len(args):
            output_file = args[idx + 1]
    elif '-o' in args:
        idx = args.index('-o')
        if idx + 1 < len(args):
            output_file = args[idx + 1]

    # Get target path (first non-flag argument not consumed by an option)
    filepath = None
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ('--output', '-o'):
            skip_next = True
            continue
        if a == 'audit':
            continue
        if not a.startswith('-'):
            filepath = a
            break
    
    if not filepath:
        filepath = "."
    
    if not os.path.exists(filepath):
        print(f"Error: Path not found: {filepath}")
        sys.exit(1)

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
        if not is_json and not is_gitlab:
            print(f"No Python (.py) or notebook (.ipynb) files found in {filepath}")
        sys.exit(0)

    total_errors = 0
    total_warnings = 0
    files_with_issues = 0
    all_json_results = {}
    gitlab_issues = []

    for f_path in files_to_scan:
        try:
            diagnostics, source_lines = analyze_file(f_path)
        except SyntaxError as e:
            if not is_json and (not is_gitlab or output_file):
                print(f"Syntax error in {f_path}: {e}")
            total_errors += 1
            continue
        except Exception as e:
            if not is_json and (not is_gitlab or output_file):
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
            all_json_results[f_path] = diagnostics
            if is_gitlab:
                rel_path = os.path.relpath(f_path).replace("\\", "/")
                for d in diagnostics:
                    sev_map = {
                        'error': 'critical',
                        'warning': 'major',
                        'info': 'info'
                    }
                    gl_sev = sev_map.get(d.get('severity', 'warning'), 'major')
                    fp = hashlib.md5(f"{rel_path}:{d.get('line', 1)}:{d.get('code', 'P000')}".encode('utf-8')).hexdigest()
                    gitlab_issues.append({
                        "description": f"[{d.get('code', 'PYCHECK')}] {d.get('message', '')}. Guarded at compile-time in NEURON: https://github.com/neuronlabs-ai/neuron-lang",
                        "check_name": d.get('code', 'pycheck'),
                        "fingerprint": fp,
                        "severity": gl_sev,
                        "location": {
                            "path": rel_path,
                            "lines": {
                                "begin": d.get('line', 1)
                            }
                        }
                    })
                if output_file:
                    format_output(f_path, diagnostics, source_lines, show_info=show_info)
            elif not is_json:
                format_output(f_path, diagnostics, source_lines, show_info=show_info)

    if is_audit:
        cert = generate_proof_of_causality_certificate(
            target_path=filepath,
            total_files=len(files_to_scan),
            total_errors=total_errors,
            total_warnings=total_warnings,
            diagnostics_by_file=all_json_results
        )
        if output_file:
            if output_file.endswith('.json'):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cert, f, indent=2)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(format_certificate_text(cert))
            print(f"Proof-of-Causality Audit Certificate written to: {output_file}\n")
        print(format_certificate_text(cert))
        print()
    elif is_gitlab:
        gl_content = json.dumps(gitlab_issues, indent=2)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(gl_content)
        else:
            print(gl_content)
    elif is_json:
        json_content = json.dumps(all_json_results, indent=2)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_content)
        else:
            print(json_content)

    # Print summary if human output or if output was redirected to a file
    if (not is_json and not is_gitlab and not is_audit) or (output_file and not is_audit):
        if len(files_to_scan) > 1 or output_file:
            clean_files = len(files_to_scan) - files_with_issues
            print("=================================================================")
            print("  PyCheck Directory Summary")
            print(f"  Scanned: {len(files_to_scan)} files | Clean: {clean_files} | Flagged: {files_with_issues}")
            print(f"  Total: {total_errors} error(s), {total_warnings} warning(s)")
            if output_file:
                report_type = "GitLab Code Quality" if is_gitlab else "JSON"
                print(f"  {report_type} Report written to: {output_file}")
            if files_with_issues > 0:
                print("-----------------------------------------------------------------")
                if total_errors > 0:
                    print(f"  These {total_errors} error(s) would be COMPILE-TIME ERRORS in NEURON")
                elif total_warnings > 0:
                    print(f"  These {total_warnings} temporal issue(s) are guarded at compile-time in NEURON")
                print("  Python detected: 0 of these issues at runtime.")
                print("  --> NEURON Zero-Leakage Language: https://github.com/neuronlabs-ai/neuron-lang")
                print("  --> PyCheck Documentation & CI/CD: https://github.com/neuronlabs-ai/pycheck")
            print("=================================================================\n")

    if total_errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

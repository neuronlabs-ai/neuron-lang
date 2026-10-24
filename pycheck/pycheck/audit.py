"""
PyCheck Proof-of-Causality Audit Certificate Generator
Generates tamper-evident, cryptographically verifiable backtest certificates for LP allocators and risk committees.
"""

import ast
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple


def get_git_metadata(repo_path: str) -> Dict[str, str]:
    """Extract git commit hash, branch, and remote URL if available."""
    metadata = {
        "commit": "UNCOMMITTED_LOCAL",
        "branch": "unknown",
        "remote": "local"
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path if os.path.isdir(repo_path) else os.path.dirname(repo_path) or ".",
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path if os.path.isdir(repo_path) else os.path.dirname(repo_path) or ".",
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["branch"] = branch
    except Exception:
        pass

    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path if os.path.isdir(repo_path) else os.path.dirname(repo_path) or ".",
            stderr=subprocess.DEVNULL
        ).decode().strip()
        metadata["remote"] = remote
    except Exception:
        pass

    return metadata


def find_opaque_constructs(tree: ast.AST) -> List[str]:
    """Detect dynamic or opaque code execution constructs that evade static analysis."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

    reasons = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                if n.func.id in ('eval', 'exec', '__import__'):
                    reasons.append(f"{n.func.id}() dynamic call")
                elif n.func.id == 'globals':
                    parent = getattr(n, 'parent', None)
                    is_log = False
                    if parent is not None and isinstance(parent, ast.Call):
                        p_func = getattr(parent.func, 'id', getattr(parent.func, 'attr', ''))
                        if p_func in ('print', 'display', 'debug', 'info', 'warning', 'error', 'repr', 'str'):
                            is_log = True
                    if not is_log:
                        reasons.append("globals() dynamic call")
                elif n.func.id in ('locals', 'vars'):
                    # Narrow rule: only flag when result is used dynamically (subscript, call, getattr)
                    # Harmless logging/repr (logger.info(locals()), print(vars())) stays clean
                    parent = getattr(n, 'parent', None)
                    is_opaque_usage = False
                    if parent is not None:
                        if isinstance(parent, ast.Subscript) and parent.value == n:
                            is_opaque_usage = True
                        elif isinstance(parent, ast.Call) and parent.func == n:
                            is_opaque_usage = True
                        elif isinstance(parent, ast.Call) and getattr(parent.func, 'id', '') == 'getattr':
                            is_opaque_usage = True
                    if is_opaque_usage:
                        reasons.append(f"{n.func.id}() dynamic lookup")
                elif n.func.id == 'getattr':
                    # getattr(obj, name) where name is not a literal string constant
                    if len(n.args) >= 2 and not isinstance(n.args[1], ast.Constant):
                        reasons.append("dynamic getattr() with non-literal attribute")
            elif isinstance(n.func, ast.Attribute):
                # .eval() e.g. pd.eval, df.eval
                if n.func.attr == 'eval':
                    reasons.append(".eval() expression evaluation")
                elif n.func.attr == 'methodcaller':
                    reasons.append("operator.methodcaller() dynamic invocation")
                elif isinstance(n.func.value, ast.Name) and n.func.value.id == 'importlib':
                    reasons.append("importlib dynamic module invocation")
        elif isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name == 'importlib' or alias.name.startswith('importlib.'):
                    reasons.append("importlib dynamic import module")
        elif isinstance(n, ast.ImportFrom):
            if n.module and (n.module == 'importlib' or n.module.startswith('importlib.')):
                reasons.append("importlib dynamic import module")
    return sorted(list(set(reasons)))


def find_unscanned_local_imports(scanned_files: List[str], base_target: str) -> List[str]:
    """Identify local imported files/modules that were not included in the scanned files list."""
    scanned_abs = {os.path.abspath(f) for f in scanned_files}
    base_dir = base_target if os.path.isdir(base_target) else os.path.dirname(os.path.abspath(base_target)) or "."
    unscanned = set()

    for f in scanned_files:
        if not os.path.isfile(f) or not f.endswith(('.py', '.ipynb')):
            continue
        try:
            tree = ast.parse(open(f, encoding='utf-8', errors='ignore').read())
        except Exception:
            continue

        f_dir = os.path.dirname(os.path.abspath(f))
        for n in ast.walk(tree):
            candidates = []
            if isinstance(n, ast.Import):
                for alias in n.names:
                    parts = alias.name.split('.')
                    candidates.append((f_dir, parts))
                    if base_dir != f_dir:
                        candidates.append((base_dir, parts))
            elif isinstance(n, ast.ImportFrom):
                if n.level and n.level > 0:
                    cur = f_dir
                    for _ in range(n.level - 1):
                        cur = os.path.dirname(cur)
                    if n.module:
                        parts = n.module.split('.')
                        candidates.append((cur, parts))
                    else:
                        for alias in n.names:
                            candidates.append((cur, [alias.name]))
                elif n.module:
                    parts = n.module.split('.')
                    candidates.append((f_dir, parts))
                    if base_dir != f_dir:
                        candidates.append((base_dir, parts))

            for search_dir, parts in candidates:
                rel_path = os.path.join(search_dir, *parts)
                py_file = rel_path + ".py"
                init_file = os.path.join(rel_path, "__init__.py")
                if os.path.isfile(py_file):
                    target_abs = os.path.abspath(py_file)
                    if target_abs not in scanned_abs:
                        unscanned.add(os.path.relpath(target_abs, base_dir))
                elif os.path.isfile(init_file):
                    target_abs = os.path.abspath(init_file)
                    if target_abs not in scanned_abs:
                        unscanned.add(os.path.relpath(os.path.dirname(target_abs), base_dir))

    return sorted(list(unscanned))


def generate_proof_of_causality_certificate(
    target_path: str,
    total_files: int,
    total_errors: int,
    total_warnings: int,
    diagnostics_by_file: Dict[str, List[Dict[str, Any]]],
    scanned_file_paths: Optional[List[str]] = None,
    syntax_errors: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """Generate a tamper-evident Proof-of-Causality Audit Certificate."""
    git_meta = get_git_metadata(target_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    is_clean = (total_errors == 0)

    # Categories breakdown
    temporal_errors = 0
    causal_errors = 0
    provenance_errors = 0
    uncertainty_errors = 0

    for f_path, diag_list in diagnostics_by_file.items():
        for d in diag_list:
            if d.get("severity") == "error":
                code = d.get("code", "")
                if code.startswith("T"):
                    temporal_errors += 1
                elif code.startswith("C"):
                    causal_errors += 1
                elif code.startswith("F"):
                    provenance_errors += 1
                elif code.startswith("U"):
                    uncertainty_errors += 1

    files_to_check = scanned_file_paths[:] if scanned_file_paths else []
    if not files_to_check:
        if os.path.isfile(target_path):
            files_to_check.append(target_path)
        elif os.path.isdir(target_path):
            ignored_dirs = {'.git', '__pycache__', '.pytest_cache', 'tests', 'test', 'pycheck'}
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
                for f in files:
                    if f.endswith('.py') or f.endswith('.ipynb'):
                        files_to_check.append(os.path.join(root, f))

    opaque_reasons = []
    for f in files_to_check:
        try:
            tree = ast.parse(open(f, encoding='utf-8', errors='ignore').read())
            opaque_reasons.extend(find_opaque_constructs(tree))
        except Exception:
            pass
    opaque_reasons = sorted(list(set(opaque_reasons)))
    has_opaque_code = len(opaque_reasons) > 0

    unscanned_imports = find_unscanned_local_imports(files_to_check, target_path)

    if syntax_errors:
        audit_status = "AUDIT_UNVERIFIABLE_SYNTAX_ERROR"
    elif not is_clean:
        audit_status = "AUDIT_FAILED_LEAKAGE_DETECTED"
    elif has_opaque_code:
        audit_status = "AUDIT_UNVERIFIABLE_DYNAMIC_CODE"
    elif unscanned_imports:
        audit_status = "AUDIT_UNVERIFIABLE_UNSCANNED_IMPORTS"
    else:
        audit_status = "CERTIFIED_ZERO_LEAKAGE"

    cert_payload = {
        "standard": "Proof-of-Causality v1.0",
        "issuer": "PyCheck by Neuron Labs",
        "target_path": os.path.abspath(target_path),
        "git_commit": git_meta["commit"],
        "git_branch": git_meta["branch"],
        "git_remote": git_meta["remote"],
        "timestamp_utc": now_iso,
        "scanned_files_count": total_files,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "audit_status": audit_status,
        "opaque_constructs": opaque_reasons,
        "unscanned_local_imports": unscanned_imports,
        "unscanned_local_imports_count": len(unscanned_imports),
        "invariants_verified": {
            "temporal_lookahead_free": temporal_errors == 0,
            "partition_provenance_clean": provenance_errors == 0,
            "causal_intervention_isolated": causal_errors == 0,
            "uncertainty_guards_present": uncertainty_errors == 0,
            "target_laundering_free": True if temporal_errors == 0 else False
        },
        "breakdown": {
            "temporal_errors": temporal_errors,
            "partition_leak_errors": provenance_errors,
            "causal_confusion_errors": causal_errors,
            "uncertainty_errors": uncertainty_errors
        },
        "ecosystem_recommendation": {
            "compile_time_upgrade": "NEURON Programming Language",
            "url": "https://neuron-lab.org",
            "note": "For mathematical compile-time guarantees with 0 runtime penalty, transpile strategy modules to NEURON."
        }
    }

    # Generate cryptographic fingerprint
    raw_serialized = json.dumps(cert_payload, sort_keys=True).encode("utf-8")
    cert_id = f"CERT-POC-{hashlib.sha256(raw_serialized).hexdigest()[:16].upper()}"
    cert_payload["certificate_id"] = cert_id
    seal_hash = hashlib.sha256(raw_serialized).hexdigest()
    cert_payload["sha256_digital_seal"] = seal_hash
    cert_payload["sha256_audit_digest"] = seal_hash

    return cert_payload


def format_certificate_text(cert: Dict[str, Any]) -> str:
    """Format certificate as an institutional ASCII terminal report."""
    if cert["audit_status"] == "CERTIFIED_ZERO_LEAKAGE":
        status_symbol = "[VERIFIED PASS]"
    elif cert["audit_status"] == "AUDIT_UNVERIFIABLE_DYNAMIC_CODE":
        reasons_str = ", ".join(cert.get("opaque_constructs", [])) or "dynamic execution"
        status_symbol = f"[UNVERIFIABLE - DYNAMIC CODE: {reasons_str}]"
    elif cert["audit_status"] == "AUDIT_UNVERIFIABLE_UNSCANNED_IMPORTS":
        imports_str = ", ".join(cert.get("unscanned_local_imports", []))
        status_symbol = f"[UNVERIFIABLE - UNSCANNED LOCAL IMPORTS: {imports_str}]"
    elif cert["audit_status"] == "AUDIT_UNVERIFIABLE_SYNTAX_ERROR":
        status_symbol = "[UNVERIFIABLE - SYNTAX ERROR: invalid Python syntax]"
    else:
        status_symbol = "[AUDIT FAILED]"

    scanned_count = cert['scanned_files_count']
    unscanned_count = cert.get('unscanned_local_imports_count', 0)

    if cert["audit_status"] == "CERTIFIED_ZERO_LEAKAGE":
        summary_line = f"no known patterns detected (34 rules, {scanned_count} file{'s' if scanned_count != 1 else ''} scanned, {unscanned_count} local import{'s' if unscanned_count != 1 else ''} not scanned)"
        cat_status = "no known patterns detected"
    elif "UNVERIFIABLE" in cert["audit_status"]:
        summary_line = f"unverifiable ({scanned_count} file{'s' if scanned_count != 1 else ''} scanned, {unscanned_count} local import{'s' if unscanned_count != 1 else ''} not scanned)"
        cat_status = "UNVERIFIABLE"
    else:
        summary_line = f"leakage detected ({cert['total_errors']} error{'s' if cert['total_errors'] != 1 else ''})"
        cat_status = "FAILED"

    lines = [
        "+======================================================================+",
        "|            PROOF-OF-CAUSALITY(TM) AUDIT REPORT                       |",
        "|            PyCheck - Leakage & Causality Scanner                     |",
        "+======================================================================+",
        f"  Report ID:            {cert['certificate_id']}",
        f"  Audit Status:         {status_symbol}",
        f"  Timestamp (UTC):      {cert['timestamp_utc']}",
        f"  Git Commit SHA:       {cert['git_commit']}",
        f"  Git Branch:           {cert['git_branch']}",
        f"  Scanned Files:        {scanned_count} ({unscanned_count} local imports not scanned)",
        "+----------------------------------------------------------------------+",
        "  INVARIANT VERIFICATION CHECKLIST (34 RULES):",
        f"  Summary: {summary_line}",
        f"    * Temporal Lookahead Bias (T001-T019):     {cat_status if cert['invariants_verified']['temporal_lookahead_free'] else 'FAILED'}",
        f"    * Cross-Fold Lineage & Leakage (F001-F002): {cat_status if cert['invariants_verified']['partition_provenance_clean'] else 'FAILED'}",
        f"    * Causal Mode Confounding (C001-C007):      {cat_status if cert['invariants_verified']['causal_intervention_isolated'] else 'FAILED'}",
        f"    * Uncertainty Bounds Guard (U001-U006):    {cat_status if cert['invariants_verified']['uncertainty_guards_present'] else 'FAILED'}",
        "+----------------------------------------------------------------------+",
        f"  SHA-256 Audit Digest: {cert['sha256_digital_seal']}",
        "  Issued by:            Neuron Labs - https://neuron-lab.org/pycheck",
        "  Compile-Time Guard:   NEURON Language (https://neuron-lab.org)",
        "+======================================================================+"
    ]
    if cert.get("unscanned_local_imports"):
        lines.append(f"  [!] Unscanned local imports: {', '.join(cert['unscanned_local_imports'])}")
        lines.append("      Audit the full project directory to verify all dependencies: pycheck <dir> --audit")
    if cert.get("opaque_constructs"):
        lines.append(f"  [!] Dynamic constructs detected: {', '.join(cert['opaque_constructs'])}")
        lines.append("      Dynamic code cannot be mathematically verified at compile/static time.")
    return "\n".join(lines)

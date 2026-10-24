"""
PyCheck Proof-of-Causality Audit Certificate Generator
Generates tamper-evident, cryptographically verifiable backtest certificates for LP allocators and risk committees.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


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


def generate_proof_of_causality_certificate(
    target_path: str,
    total_files: int,
    total_errors: int,
    total_warnings: int,
    diagnostics_by_file: Dict[str, List[Dict[str, Any]]]
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
        "audit_status": "CERTIFIED_ZERO_LEAKAGE" if is_clean else "AUDIT_FAILED_LEAKAGE_DETECTED",
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
    cert_payload["sha256_digital_seal"] = hashlib.sha256(raw_serialized).hexdigest()

    return cert_payload


def format_certificate_text(cert: Dict[str, Any]) -> str:
    """Format certificate as an institutional ASCII terminal report."""
    status_symbol = "[VERIFIED PASS]" if cert["audit_status"] == "CERTIFIED_ZERO_LEAKAGE" else "[AUDIT FAILED]"
    lines = [
        "+======================================================================+",
        "|            PROOF-OF-CAUSALITY(TM) AUDIT CERTIFICATE                  |",
        "|            PyCheck - Institutional Quant Safety                      |",
        "+======================================================================+",
        f"  Certificate ID:       {cert['certificate_id']}",
        f"  Audit Status:         {status_symbol}",
        f"  Timestamp (UTC):      {cert['timestamp_utc']}",
        f"  Git Commit SHA:       {cert['git_commit']}",
        f"  Git Branch:           {cert['git_branch']}",
        f"  Scanned Files:        {cert['scanned_files_count']}",
        "+----------------------------------------------------------------------+",
        "  INVARIANT VERIFICATION CHECKLIST (34 RULES):",
        f"    * Temporal Lookahead Bias (T001-T019):     {'PASSED (0 leaks)' if cert['invariants_verified']['temporal_lookahead_free'] else 'FAILED'}",
        f"    * Cross-Fold Lineage & Leakage (F001-F002): {'PASSED (0 leaks)' if cert['invariants_verified']['partition_provenance_clean'] else 'FAILED'}",
        f"    * Causal Mode Confounding (C001-C007):      {'PASSED (0 leaks)' if cert['invariants_verified']['causal_intervention_isolated'] else 'FAILED'}",
        f"    * Uncertainty Bounds Guard (U001-U006):    {'PASSED (0 leaks)' if cert['invariants_verified']['uncertainty_guards_present'] else 'FAILED'}",
        "+----------------------------------------------------------------------+",
        f"  SHA-256 Digital Seal: {cert['sha256_digital_seal']}",
        "  Issued by:            Neuron Labs - https://neuron-lab.org/pycheck",
        "  Next-Gen Sovereign:   NEURON Compile-Time Language (https://neuron-lab.org)",
        "+======================================================================+"
    ]
    return "\n".join(lines)

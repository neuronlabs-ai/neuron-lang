"""
Unit tests for PyCheck Proof-of-Causality Audit Certificate generator.
"""

import os
import json
from pycheck.audit import generate_proof_of_causality_certificate, format_certificate_text


def test_audit_certificate_clean():
    cert = generate_proof_of_causality_certificate(
        target_path=".",
        total_files=5,
        total_errors=0,
        total_warnings=2,
        diagnostics_by_file={}
    )
    assert cert["audit_status"] == "CERTIFIED_ZERO_LEAKAGE"
    assert cert["invariants_verified"]["temporal_lookahead_free"] is True
    assert cert["certificate_id"].startswith("CERT-POC-")
    assert len(cert["sha256_digital_seal"]) == 64
    
    text = format_certificate_text(cert)
    assert "[VERIFIED PASS]" in text
    assert cert["certificate_id"] in text


def test_audit_certificate_leaks_detected():
    fake_diags = {
        "strategy.py": [
            {"code": "T001", "severity": "error", "message": "Future shift leak"},
            {"code": "F001", "severity": "error", "message": "Cross-fold leakage"}
        ]
    }
    cert = generate_proof_of_causality_certificate(
        target_path="strategy.py",
        total_files=1,
        total_errors=2,
        total_warnings=0,
        diagnostics_by_file=fake_diags
    )
    assert cert["audit_status"] == "AUDIT_FAILED_LEAKAGE_DETECTED"
    assert cert["invariants_verified"]["temporal_lookahead_free"] is False
    assert cert["invariants_verified"]["partition_provenance_clean"] is False
    assert cert["breakdown"]["temporal_errors"] == 1
    assert cert["breakdown"]["partition_leak_errors"] == 1
    
    text = format_certificate_text(cert)
    assert "[AUDIT FAILED]" in text

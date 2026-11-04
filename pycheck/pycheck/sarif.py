"""
SARIF v2.1.0 generator for PyCheck.
Converts PyCheck diagnostics into standard OASIS SARIF v2.1.0 JSON format
for GitHub Code Scanning, VS Code SARIF Viewer, and CI/CD integrations.
"""

import os
from typing import Dict, List, Any, Optional
from pycheck.rules import ALL_RULES, Rule


def _rule_to_sarif(rule: Rule) -> Dict[str, Any]:
    sev_map = {
        "error": "error",
        "warning": "warning",
        "info": "note",
    }
    desc = (rule.__doc__ or "").strip()
    if not desc:
        desc = rule.name.replace("_", " ").capitalize()
    first_line = desc.split("\n")[0]

    return {
        "id": rule.code,
        "name": rule.name,
        "shortDescription": {"text": first_line},
        "fullDescription": {"text": desc},
        "defaultConfiguration": {
            "level": sev_map.get(rule.severity, "warning")
        },
        "helpUri": "https://github.com/neuronlabs-ai/neuron-lang/tree/main/pycheck#rule-catalog",
    }


def generate_sarif(
    diagnostics_by_file: Dict[str, List[Dict[str, Any]]],
    rules: Optional[List[Rule]] = None,
    version: str = "1.2.0",
) -> Dict[str, Any]:
    """
    Generate a SARIF v2.1.0 log document from PyCheck diagnostics.
    """
    if rules is None:
        rules = ALL_RULES

    sarif_rules = [_rule_to_sarif(r) for r in rules]

    sarif_results = []
    sev_map = {
        "error": "error",
        "warning": "warning",
        "info": "note",
    }

    for file_path, diags in sorted(diagnostics_by_file.items()):
        rel_path = os.path.relpath(file_path).replace("\\", "/")
        for d in diags:
            code = d.get("code", "PYCHECK")
            severity = d.get("severity", "warning")
            message = d.get("message", "")
            suggestion = d.get("suggestion", "")
            line = d.get("line", 1)
            col = max(1, d.get("col", 0) + 1)

            full_msg = message
            if suggestion:
                full_msg += f". Suggestion: {suggestion}"

            sarif_results.append({
                "ruleId": code,
                "level": sev_map.get(severity, "warning"),
                "message": {
                    "text": full_msg
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": rel_path,
                                "uriBaseId": "%SRCROOT%"
                            },
                            "region": {
                                "startLine": line,
                                "startColumn": col
                            }
                        }
                    }
                ]
            })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "PyCheck",
                        "semanticVersion": version,
                        "informationUri": "https://github.com/neuronlabs-ai/neuron-lang/tree/main/pycheck",
                        "rules": sarif_rules,
                    }
                },
                "results": sarif_results,
            }
        ],
    }


def format_github_action_annotation(file_path: str, diag: Dict[str, Any]) -> str:
    """Format a single diagnostic as a GitHub Actions workflow annotation."""
    sev = diag.get("severity", "warning")
    cmd = "error" if sev == "error" else ("warning" if sev == "warning" else "notice")
    rel_path = os.path.relpath(file_path).replace("\\", "/")
    line = diag.get("line", 1)
    col = max(1, diag.get("col", 0) + 1)
    code = diag.get("code", "PYCHECK")
    msg = diag.get("message", "")
    suggestion = diag.get("suggestion", "")
    full_msg = f"{msg} | Suggestion: {suggestion}" if suggestion else msg
    # GitHub actions workflow command escaping: newlines to %0A, % to %25, \r to %0D
    full_msg = full_msg.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::{cmd} file={rel_path},line={line},col={col},title={code}::{full_msg}"

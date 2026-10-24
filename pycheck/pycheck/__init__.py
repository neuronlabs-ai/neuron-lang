"""
PyCheck — NEURON ML Safety Analyzer
Catches temporal leaks, causal confusion, and uncertainty bugs in Python ML scripts.
"""

__version__ = "1.0.7"

from pycheck.analyzer import analyze_file, analyze_code, format_output
from pycheck.rules import ALL_RULES, RULES_BY_CODE
from pycheck.flow import TaintTracker

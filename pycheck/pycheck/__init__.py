"""
PyCheck — NEURON Python Safety Analyzer
Scans Python ML and trading scripts for temporal lookahead bias,
causal confusion, and unguarded uncertainty bugs.
"""

from pycheck.analyzer import NeuronAnalyzer, analyze_file, format_output

__version__ = "0.1.0"
__all__ = ["NeuronAnalyzer", "analyze_file", "format_output"]

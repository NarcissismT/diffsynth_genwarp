"""Evaluation visualization lives in evaluate_full to share exact tensors.

The public module exists as the stable ownership location for future panel or
heatmap extensions; no training code should import evaluator-private helpers.
"""


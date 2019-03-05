"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .anchor.anchor_tabular import AnchorTabular

__all__ = ["AnchorTabular"]

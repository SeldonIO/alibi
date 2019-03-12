"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .anchor.anchor_tabular import AnchorTabular
from .anchor.anchor_text import AnchorText

__all__ = ["AnchorTabular",
           "AnchorText"]

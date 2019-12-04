"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .anchor_tabular import AnchorTabular, _show_anchortabular
from .anchor_text import AnchorText
from .anchor_image import AnchorImage
from .cem import CEM
from .cfproto import CounterFactualProto
from .counterfactual import CounterFactual

__all__ = ["AnchorTabular",
           "AnchorText",
           "AnchorImage",
           "CEM",
           "CounterFactual",
           "CounterFactualProto"]

_VIZFUNS = [_show_anchortabular,
            None,
            None,
            None,
            None,
            None]

# dictionary for dispatching visualizations
DISPATCH_DICT = dict(zip(__all__, _VIZFUNS))

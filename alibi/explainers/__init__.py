"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .ale import ALE, plot_ale
from .anchor_tabular import AnchorTabular, DistributedAnchorTabular
from .anchor_text import AnchorText
from .anchor_image import AnchorImage
from .cem import CEM
from .cfproto import CounterFactualProto
from .counterfactual import CounterFactual
from .integrated_gradients import IntegratedGradients
from .cfrl_base import CounterfactualRLBase
from .cfrl_tabular import CounterfactualRLTabular

__all__ = ["ALE",
           "AnchorTabular",
           "DistributedAnchorTabular",
           "AnchorText",
           "AnchorImage",
           "CEM",
           "CounterFactual",
           "CounterFactualProto",
           "CounterfactualRLBase",
           "CounterfactualRLTabular",
           "plot_ale",
           "IntegratedGradients",
           ]

try:
    from .shap_wrappers import KernelShap, TreeShap
    __all__ += ["KernelShap", "TreeShap"]
except ImportError:
    pass

"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .ale import ALE, plot_ale
from .anchor_tabular import AnchorTabular, DistributedAnchorTabular
from .anchor_text import AnchorText
from .anchor_image import AnchorImage
from .cem import CEM
from .cfproto import CounterfactualProto, CounterFactualProto  # noqa: F401 TODO: remove in an upcoming release
from .counterfactual import Counterfactual, CounterFactual  # noqa: F401 TODO: remove in an upcoming release
from .integrated_gradients import IntegratedGradients
from .cfrl_base import CounterfactualRL
from .cfrl_tabular import CounterfactualRLTabular

__all__ = ["ALE",
           "AnchorTabular",
           "DistributedAnchorTabular",
           "AnchorText",
           "AnchorImage",
           "CEM",
           "Counterfactual",
           "CounterfactualProto",
           "CounterfactualRL",
           "CounterfactualRLTabular",
           "plot_ale",
           "IntegratedGradients",
           ]

try:
    from .shap_wrappers import KernelShap, TreeShap

    __all__ += ["KernelShap", "TreeShap"]
except ImportError:
    pass

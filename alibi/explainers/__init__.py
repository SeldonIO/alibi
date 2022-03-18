"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from alibi.utils.missing_optional_dependency import import_optional

from .ale import ALE, plot_ale
from .anchor_tabular import AnchorTabular

DistributedAnchorTabular = import_optional(
    'alibi.explainers.anchor_tabular_distributed',
    names=['DistributedAnchorTabular'])

from .anchor_text import AnchorText
from .anchor_image import AnchorImage
from .cem import CEM
from .cfproto import CounterfactualProto, CounterFactualProto  # noqa: F401 TODO: remove in an upcoming release
from .counterfactual import Counterfactual, CounterFactual  # noqa: F401 TODO: remove in an upcoming release
from .integrated_gradients import IntegratedGradients
from .cfrl_base import CounterfactualRL
from .cfrl_tabular import CounterfactualRLTabular

KernelShap = import_optional(
    'alibi.explainers.shap_wrappers',
    names=['KernelShap'])

TreeShap = import_optional(
    'alibi.explainers.shap_wrappers',
    names=['TreeShap'])

__all__ = [
    "ALE",
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
    "KernelShap",
    "TreeShap"
]

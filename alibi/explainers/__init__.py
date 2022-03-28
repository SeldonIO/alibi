"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from alibi.utils.missing_optional_dependency import import_optional
from .ale import ALE, plot_ale
from alibi.explainers.anchors.anchor_text import AnchorText
from alibi.explainers.anchors.anchor_tabular import AnchorTabular
from alibi.explainers.anchors.anchor_image import AnchorImage
from .cfrl_base import CounterfactualRL, Callback
from .cfrl_tabular import CounterfactualRLTabular


DistributedAnchorTabular = import_optional(
    'alibi.explainers.anchors.anchor_tabular_distributed',
    names=['DistributedAnchorTabular'])

CEM = import_optional(
    'alibi.explainers.cem',
    names=['CEM'])

CounterfactualProto, CounterFactualProto = import_optional(
    'alibi.explainers.cfproto',
    names=['CounterfactualProto', 'CounterFactualProto'])  # TODO: remove in an upcoming release

Counterfactual, CounterFactual = import_optional(
    'alibi.explainers.counterfactual',
    names=['Counterfactual', 'CounterFactual'])  # TODO: remove in an upcoming release

IntegratedGradients = import_optional(
    'alibi.explainers.integrated_gradients',
    names=['IntegratedGradients'])

KernelShap, TreeShap, sum_categories = import_optional(
    'alibi.explainers.shap_wrappers',
    names=['KernelShap', 'TreeShap', 'sum_categories'])

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
    "Callback",
    "plot_ale",
    "IntegratedGradients",
    "KernelShap",
    "TreeShap"
]

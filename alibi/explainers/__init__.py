"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from alibi.utils.missing_optional_dependency import MissingOptionalDependency

from .ale import ALE, plot_ale
from .anchor_tabular import AnchorTabular

try:
    from .anchor_tabular_distributed import DistributedAnchorTabular
except ModuleNotFoundError as err:
    DistributedAnchorTabular = MissingOptionalDependency(err, "DistributedAnchorTabular", install_option='ray')

from .anchor_text import AnchorText
from .anchor_image import AnchorImage
from .cem import CEM
from .cfproto import CounterfactualProto, CounterFactualProto  # noqa: F401 TODO: remove in an upcoming release
from .counterfactual import Counterfactual, CounterFactual  # noqa: F401 TODO: remove in an upcoming release
from .integrated_gradients import IntegratedGradients
from .cfrl_base import CounterfactualRL
from .cfrl_tabular import CounterfactualRLTabular

try:
    from .shap_wrappers import KernelShap, TreeShap
except ModuleNotFoundError as err:
    KernelShap = MissingOptionalDependency(err, "KernelShap", install_option='shap')
    TreeShap = MissingOptionalDependency(err, "TreeShap", install_option='shap')


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

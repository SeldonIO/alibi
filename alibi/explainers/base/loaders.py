# flake8: noqa: F401

from typing import Optional
from alibi.explainers.base import explainer_registry


def get_implementation(explainer_type: str, method: Optional[str] = None):
    """
    Returns a class that implements an explainer from the explainer registry.

    Paramters
    --------
    explainer_type
        The explanation algorithm type.
    method
        Indicates a specific implementation of an explainer type.
    """
    if explainer_type == 'counterfactual':
        return get_counterfactual_implementation(method)


def get_counterfactual_implementation(method: str):
    """
    Returns a counterfactual implementation from the explainer registry.

    Parameters
    ----------
    method
        Indicates which counterfactual method should be returned.
    """

    # import statement has the effect of updating the registry
    import alibi.explainers.base.counterfactuals
    if method == 'wachter':
        return explainer_registry['counterfactual'][method]
    raise ValueError(f"Method {method} not implemented!")

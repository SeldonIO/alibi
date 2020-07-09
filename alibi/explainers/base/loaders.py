# flake8: noqa: F401

from typing import Optional
from alibi.explainers.base import explainer_registry


def get_implementation(explainer_type: str, method: Optional[str] = None):

    # TODO: ALEX: DOCSTRING
    if explainer_type == 'counterfactual':
        return get_counterfactual_implementation(method)


def get_counterfactual_implementation(method: str):

    # TODO: ALEX: DOCSTRING
    # import statement has the effect of updating the registry
    import alibi.explainers.base.counterfactuals
    if method == 'wachter':
        return explainer_registry['counterfactual'][method]
    raise ValueError(f"Method {method} not implemented!")

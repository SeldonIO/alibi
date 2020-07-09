# flake8: noqa: F401
from typing import Optional
from alibi.explainers.backend import backends_registry


def load_backend(explainer_type: str, framework: str, predictor_type: str, method: Optional[str] = None):
    """
    A dispatch function that delegates the loading of the explainer to specialised functions for each explainer type.

    Parameters
    ----------
    explainer_type: {'countefactual', 'cem', 'attribution'}
        Indicates the type of the explanation algorithm loaded.
    framework: {'pytorch', 'tensorflow'}
        Indicates which backend should be used.
    predictor_type: {'blackbox', 'whitebox'}
        Indicates whether the explainer has access to model parameters.
    method
        Indicates the specific implementation for the explainer type (e.g., for counterfactual algorithms 'wachter'
        means that the backend for the algorithm proposed by `Wachter et al. (2017)`_ (pp. 854) will be loaded).
        
        .. _Wachter et al. (2017): 
           https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf
    """  # noqa W605

    if explainer_type == 'counterfactual':
        return load_counterfactual(framework, predictor_type, method)
    elif explainer_type == 'cem':
        return load_cem(framework, predictor_type, method)
    elif explainer_type == 'attribution':
        return load_attribution(framework, predictor_type, method)
    else:
        raise ValueError(f"Unknown explainer type {explainer_type}")


def load_counterfactual(framework: str, predictor_type: str, method: str):
    """
    A loading function that imports the modules where counterfactual algorithms are implemented in order to add the
    desired framework-specific counterfactual implementation to the explainer registry and then return it.

    Parameters
    ----------
    framework, predictor_type, method:
        See `load_backend` function.
    """

    # TODO: ALEX: TBD: __init__ alternative

    if framework == 'tensorflow':
        # import so that the implementation is registered
        import alibi.explainers.backend.tensorflow.counterfactuals
    else:
        import alibi.explainers.backend.pytorch.counterfactual
        print("backends_registry", backends_registry)

    return backends_registry['counterfactual'][method][framework][predictor_type]


def load_cem(framework: str, predictor_type: str, method: str):
    raise NotImplementedError


def load_attribution(framework: str, predictor_type: str, method: str):
    raise NotImplementedError

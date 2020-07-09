from collections import defaultdict
from typing import Optional


def method_dict_factory():
    """
    A factory function that allows to define implementations for a given method
    """
    method_spec = {
        'pytorch': {'blackbox': None, 'whitebox': None},
        'tensorflow': {'blackbox': None, 'whitebox': None},
    }
    return method_spec


backends_registry = {
    # we have different methods to search for a cf (e.g., watcher, proto)
    'counterfactual': {
        'wachter': defaultdict(method_dict_factory),
        'proto': defaultdict(method_dict_factory)
    },
    # no methods for CEM, but the registry decorator should still be able to register a method if we want to
    'cem': method_dict_factory(),
    'attribution': method_dict_factory(),
}


def register_backend(explainer_type: str, predictor_type: str, method: Optional[str] = None):
    """
    A parametrized decorator that can be used to register a PyTorch or TensorFlow specific explainer implementations.
    The decorator is used to access the implementations in various modules in the `alibi.explainers.backend` package.
    This decorator expects the registered class to have an attribute `framework_name`, set to either 'tensorflow' or
    'pytorch'.

    Parameters
    ----------
    explainer_type: {'countefactual', 'cem', 'attribution'}
        Indicates the type of the explanation algorithm loaded.
    predictor_type: {'blackbox', 'whitebox'}
        Indicates whether the registered backend has access to the predictor parameters.
    method
        If passed, the backend registered is a specific implementation of a particular explanation type. For example,
        a counterfactual explainer admits 'proto' as a method, implementing counterfactual search guided by prototypes.
    """

    if explainer_type not in backends_registry:
        raise ValueError(
            f"Only backends for explainers of type {backends_registry.keys()} can be registered. Update the registry "
            f"with a new type if you wish to register a new explainer type!"
        )
    if predictor_type not in ['blackbox', 'whitebox']:
        raise ValueError(f"Predictor type must be {predictor_type}")

    backends = backends_registry[explainer_type]

    def register(obj):
        if not hasattr(obj, 'framework_name'):
            raise ValueError(
                "To register a backend, the class should have an attribute 'framework_name' with value 'tensorflow'"
                " or 'pytorch'"
            )

        if method:
            if isinstance(list(backends.keys())[0], str):
                backends[method][obj.framework_name][predictor_type] = obj
            else:  # an algo for which we don't have any methods registered, just one backend
                backends[method] = method_dict_factory()
                backends[method][obj.framework_name][predictor_type] = obj
        else:
            backends[obj.framework_name][predictor_type] = obj
        return obj
    return register

from collections import defaultdict
from typing_extensions import Literal
"""
A module that implements a registry containing framework-specific explainer implementaions along with a decorator to 
update it. 
        
The registry is updated by decorating the backend classes with the `register_backend` parameterized decorator.
    
The registry should be used to return the backend class to calling objects using the 
`alibi.explainers.backend.common.load_backend` function.
""" # noqa W605

FRAMEWORKS = ['pytorch', 'tensorflow']


def framework_factory():
    """
    A factory function that returns a mapping where objects implementing black- and white-box variants of a certain
    algorithm backend can be stored.
    """
    return {'blackbox': None, 'whitebox': None}


backends_registry = {
    'pytorch': defaultdict(framework_factory),
    'tensorflow': defaultdict(framework_factory),
}
"""
The backends registry is a dictionary, structured as follows::

    {
    'pytorch': {'BackendClass': {'blackbox': None, 'whitebox': None}, ...}
    'tensorflow': {'BackendClass': {'blackbox': None, 'whitebox': None}, ...}
    }
    
    Here `BackendClass` and the framework name are specified by the user using the `register_backend` decorator. Each 
    backend class can optionally have a predictor type, `whitebox` and `blackbox`. If not specified, the predictor type
    is assumed to be `whitebox`. See `register_backend` documentation for more details.
"""


def register_backend(consumer_class: str, predictor_type: Literal['whitebox', 'blackbox'] = 'whitebox'):
    """
    A parametrized decorator that can be used to register a class that contains PyTorch or TensorFlow backend
    implementations for explainers. The decorator is used to access the implementations in various modules in the
    `alibi.explainers.backend` package via the `load_backend` function. This decorator expects the registered class to
    have an attribute `framework_name`, set to either 'tensorflow' or 'pytorch'.

    Parameters
    ----------
    consumer_class
        This should be the name of the class for which this backend is implemented.
    predictor_type: {'whitebox', 'blackbox'}
        'whitebox' indicates that the registered backend has access to the predictor parameters (e.g., it can use the
        framework autograd functions for differentiation). `blackbox` indicates that the predictor should be treated as
        a black-box (e.g., numerical differentiation is needed to differentiate the predictor output wrt its input).

    Examples
    --------
    The `TFWachterCounterfactualOptimizer` class is a TensorFlow backend for the `_WachterCounterfactual` class. Then,
    in `alibi.explainers.backend.tensorflow.counterfactual` the registration code would be as follows::

    >>> @register_backend(consumer_class='_WachterCounterfactual', framework='tensorflow')
    >>> class TFWachterCounterfactualOptimizer(TFCounterfactualOptimizer): ...

    Raises
    ------
    ValueError
        - If the predictor type value is not correct
        - If the object registred does not have a `framework_name` class attribute
        - If the `framework_name` is not correctly set
    """

    if predictor_type not in ['blackbox', 'whitebox']:
        raise ValueError(f"Predictor type must be 'blackbox' or 'whitebox' but got {predictor_type}")

    def register(obj):
        if not hasattr(obj, 'framework'):
            raise ValueError(
                f"To register a backend, the class should have an attribute 'framework_name' with one of the following "
                f"values: {FRAMEWORKS}"
            )
        framework = obj.framework
        if framework not in FRAMEWORKS:
            raise ValueError(
                f"Framework must be 'pytorch' or 'tensorflow' but got {framework}"
            )
        backends_registry[framework][consumer_class][predictor_type] = obj
        return obj
    return register

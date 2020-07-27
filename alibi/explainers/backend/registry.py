from collections import defaultdict
from typing import Optional
"""
A module that implements a backend registry along with a decorator to update it. 
        
The registry is updated by decorating the backend classes with the `register_backend` parameterized decorator. The
`explainer_type` and the `predictor_type` arguments are mandatory. `method` is optional since there might not be 
multiple methods that implement a particular `explainer_type`. 
    
The registry should be used to return the backend class using the `alibi.explainers.backend.common.load_backend` 
function.
""" # noqa W605


def framework_factory():
    """
    A factory function that returns a mapping where objects implementing black- and white-box variants of a certain
    algorithm can be stored.
    """
    return {'blackbox': None, 'whitebox': None}


def method_dict_factory():
    """
    A factory function that returns a mapping where objects implementing black- and white-box variants of a certain
    algorithm can be stored, for both PyTorch and TensorFlow.
    """
    return {'pytorch': framework_factory(), 'tensorflow': framework_factory()}


_default_registry = {
    # we have different methods to search for a cf (e.g., watcher, proto)
    'counterfactual': {
        'wachter': {'pytorch': framework_factory(), 'tensorflow': framework_factory()},
        'proto': {'pytorch': framework_factory(), 'tensorflow': framework_factory()},
    },
    # no methods for CEM, but the registry decorator should still be able to register a method if we want to
    'cem': method_dict_factory(),
    'attribution': method_dict_factory(),
}

backends_registry = defaultdict(method_dict_factory, _default_registry)
"""
The backend registry is a dictionary
that is structured as follows::
    
    {
    'explainer_type_1':   
        {
        'method_1': {
                    'pytorch': {'whitebox': None, 'blackbox': None}
                    'tensorflow': {'whitebox': None, 'blackbox': None}
                    }
        'method_2' : {
                    'pytorch': {'whitebox': None, 'blackbox': None}
                    'tensorflow': {'whitebox': None, 'blackbox': None}
                    }
        }
    'explainer_type_2':
        {
        'pytorch': {'whitebox': None, 'blackbox': None}
        'tensorflow': {'whitebox': None, 'blackbox': None}
        }
    } 
    
    Specifically:
    
        - The top level key indicates the explanation algorithm type (e.g., 'counterfactual', 'cem', 'attribution')
        - An explanation algorithm might have multiple methods (e.g., counterfactual methods would be 'wachter', 'proto', \
        'offline' defined as the next level keys
        - If an explainer type doesn't have multiple methods, then a mapping containg the framework names is the next \
        level. Each key allows storing a 'whitebox' and a 'blackbox' backend of the method/explainer type, in the \
        framework indicated by the key. 
"""


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
        If passed, the backend registered is a specific implementation of a particular explanation type. For example, a
        counterfactual explainer admits 'proto' as a method, implementing counterfactual search guided by prototypes.
    """

    # TODO: ALEX: TBD: SHOULD WE RAISE THIS ERROR?
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
            # check if, for the current explainer type, the method specified is already in the default registry
            # if the method is registred by default, then we just update the field with the object
            if isinstance(list(backends.keys())[0], str):
                backends[method][obj.framework_name][predictor_type] = obj
            # the method specified is not in the default registry. In this case, we need to create a field with the
            # method name, which maps to a dictionary as returned by `method_dict_factory()`. We update the field
            # the framework and predictor type, as before.
            else:
                backends[method] = method_dict_factory()
                backends[method][obj.framework_name][predictor_type] = obj
        # for some algorithms, "method" might not make sense. For those explainer types, the registry contains a key for
        # each framework as opposed to a key for each method that implements a particular explainer type.
        else:
            backends[obj.framework_name][predictor_type] = obj
        return obj
    return register

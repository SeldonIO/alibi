from typing import Optional

explainer_registry = {
    'counterfactual': {'wachter': None},
    'cem': None,
    'attribution': None,
}


def register_explainer(explainer_type: str, method: Optional[str] = None):
    """
    A parametrized decorator that can be used to register a new explanation algorithm (i.e., method) for an existining
    explainer type (e..g, counterfactual). Its intended use is to collect an explainers' implementation (or classes
    inheriting from it) into an explainers registry which is used by loading functions (implemented in
    `alibi.explainers.base`) to return the correct implementation when called from `Explainer` classes.

    Parameters
    ----------
    explainer_type: {'countefactual', 'cem', 'attribution'}
        Indicates the type of the explanation algorithm loaded.
    method
        If passed, the backend registered is a specific implementation of a particular explanation type. For example,
        a counterfactual explainer admits 'proto' as a method, implementing counterfactual search guided by prototypes.
    """

    if explainer_type not in explainer_registry:
        raise ValueError(
            f"Only explainers of type {explainer_registry.keys()} can be registered. Update the registry with a new "
            f"type if you wish to register a new explainer type!"
        )

    explainers = explainer_registry[explainer_type]

    def register(obj):
        if method:
            if explainers:
                explainers[method] = obj
            else:  # add a method for an existing explainer type with a single implementation
                explainer_registry[explainer_type] = {method: obj}
        else:
            explainer_registry[explainer_type] = obj
        return obj
    return register

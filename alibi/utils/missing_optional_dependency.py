"""Methods and object for optional importing

This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.

Instead of replacing the objects with an instance of a Missing Dependency class we instead replace them with a class
itself. This is done to allow for type checking throughout the codebase. This means that instead of defining a class
here we define a metaclass that is used to define the class. For instance because `MissingDependencyTensorFlow` extends
`type` and not `object` we can say:

```py
MissingDependencyTensorFlow('IntegratedGradients', (object,), {'err': err})
```

The above will be a class that explodes on use by raising the error passed via the `attrs` in the `__new__` method.
Importantly we can pass the above to typing constructs such as `Union`.

For further discussion see: https://github.com/SeldonIO/alibi/pull/583
"""


from typing import Union, List, Optional
from string import Template
from importlib import import_module

err_msg_template = Template((
    f"Attempted to use $name without the correct optional dependencies installed. To install "
    + f"the correct optional dependencies, run `pip install alibi[$missing_dependency]` "
    + "from the command line. For more information, check the 'Dependency installs' section "
    + "of the installation docs at https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html."
))


class MissingDependency(type):
    """Metaclass for Missing Dependency classes

    Replaces any object that requires unmet optional dependencies. Attribute access or attempting to initialize
    classes derived from this metaclass will raise an error.
    """
    err: Union[ModuleNotFoundError, ImportError]
    missing_dependency: str = 'all'

    def __new__(mcs, name, bases, attrs):
        return super(MissingDependency, mcs) \
            .__new__(mcs, name, bases, attrs)

    @property
    def err_msg(cls):
        return err_msg_template.substitute(
            name=cls.__name__,
            missing_dependency=cls.missing_dependency)

    def __getattr__(cls, key):
        raise ImportError(cls.err_msg) from cls.err

    def __call__(cls):
        raise ImportError(cls.err_msg) from cls.err


class MissingDependencyRay(MissingDependency):
    missing_dependency = 'ray'


class MissingDependencyTensorFlow(MissingDependency):
    missing_dependency = 'tensorflow'


class MissingDependencyTorch(MissingDependency):
    missing_dependency = 'torch'


class MissingDependencyShap(MissingDependency):
    missing_dependency = 'shap'


def import_optional(module: str, names: Optional[List[str]] = None):
    """Import a module that depends on optional dependencies

    params:
    ______
        module:
            The module to import
        names:
            The names to import from the module. If None, all names are imported.

    returns:
    ______
        The module or named objects within the modules if names is not None. If the import fails due to a
        ModuleNotFoundError or ImportError. The requested module or named objects are replaced with classes derived
        from metaclass corresponding to the relevant optional dependency in `extras_requirements`. These classes will
        have the same name as the objects that failed to import.
    """

    try:
        module = import_module(module)
        # TODO: if want to check against specific versions we should do so here.
        if names is not None:
            objs = tuple(getattr(module, name) for name in names)
            return objs if len(objs) > 1 else objs[0]
        return module
    except ModuleNotFoundError as err:
        error_type = {
            'ray': MissingDependencyRay,
            'tensorflow': MissingDependencyTensorFlow,
            'torch': MissingDependencyTorch,
            'shape': MissingDependencyShap
        }.get(err.name, MissingDependency)
        if names is not None:
            missing_dependencies = tuple(error_type(name, (object,), {'err': err}) for name in names)
            return missing_dependencies if len(missing_dependencies) > 1 else missing_dependencies[0]
        return error_type(module, (object,), {'err': err})
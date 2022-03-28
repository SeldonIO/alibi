"""Functionality for optional importing

This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.

We replace objects that require unmet dependencies with a class that generates errors on use. We use a class instead
of an instance as this allows for type checking throughout the codebase. This means that instead of defining a class
here we define a metaclass that is used to define the class we replace the object with. For instance because
`MissingDependencyTensorFlow` extends `type` and not `object` we can say:

```py
MissingDependencyTensorFlow('IntegratedGradients', (object,), {'err': err})
```

The above will be a class that raises the error passed via the `attrs` in the `__new__` method as well as informing the
user of the necessary steps to fix. Importantly we can pass the above to typing constructs such as `Union`.

For further discussion see: https://github.com/SeldonIO/alibi/pull/583
"""


from typing import Union, List, Optional
from string import Template
from importlib import import_module

err_msg_template = Template((
    "Attempted to use $name without the correct optional dependencies installed. To install "
    + "the correct optional dependencies, run `pip install alibi[$missing_dependency]` "
    + "from the command line. For more information, check the Installation documentation "
    + "at https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html."
))


class MissingDependency(type):
    """Metaclass for Missing Dependency classes

    Replaces any object that requires unmet optional dependencies. Attribute access or attempting to initialize
    classes derived from this metaclass will raise an error.
    """
    err: Union[ModuleNotFoundError, ImportError]
    missing_dependency: str = 'all'

    def __new__(mcs, name, bases, attrs):
        """ Metaclass for MissingDependency classes

        Parameters
        ----------
        name:
            Name of the class to be created
        bases:
            Base classes of the class to be created
        attrs:
            Attributes of the class to be created, should contain an `err` attribute that will be used to raise an
            error when the class is accessed or initialized.
        """
        return super(MissingDependency, mcs) \
            .__new__(mcs, name, bases, attrs)

    @property
    def err_msg(cls):
        """Generate error message informing user to install missing dependencies."""
        return err_msg_template.substitute(
            name=cls.__name__,
            missing_dependency=cls.missing_dependency)

    def __getattr__(cls, key):
        """Raise an error when attributes are accessed."""
        raise ImportError(cls.err_msg) from cls.err

    def __call__(cls):
        """Raise an error if initialized."""
        raise ImportError(cls.err_msg) from cls.err


class MissingDependencyRay(MissingDependency):
    missing_dependency = 'ray'


class MissingDependencyTensorFlow(MissingDependency):
    missing_dependency = 'tensorflow'


class MissingDependencyTorch(MissingDependency):
    missing_dependency = 'torch'


class MissingDependencyShap(MissingDependency):
    missing_dependency = 'shap'


ERROR_TYPES = {
    'ray': MissingDependencyRay,
    'tensorflow': MissingDependencyTensorFlow,
    'torch': MissingDependencyTorch,
    'shap': MissingDependencyShap,
    'numba': MissingDependencyShap,
}


def import_optional(module_name: str, names: Optional[List[str]] = None):
    """Import a module that depends on optional dependencies

    params:
    ______
        module:
            The module to import
        names:
            The names to import from the module. If None, all names are imported.

    returns:
    _______
        The module or named objects within the modules if names is not None. If the import fails due to a
        ModuleNotFoundError or ImportError. The requested module or named objects are replaced with classes derived
        from metaclass corresponding to the relevant optional dependency in `extras_requirements`. These classes will
        have the same name as the objects that failed to import but will error on use.
    """

    try:
        module = import_module(module_name)
        # TODO: We should check against specific dependency versions here.
        if names is not None:
            objs = tuple(getattr(module, name) for name in names)
            return objs if len(objs) > 1 else objs[0]
        return module
    except ModuleNotFoundError as err:
        if err.name is None:
            raise TypeError()
        if err.name not in ERROR_TYPES:
            raise err
        error_type = ERROR_TYPES[err.name]
        if names is not None:
            missing_dependencies = tuple(error_type(name, (object,), {'err': err}) for name in names)
            return missing_dependencies if len(missing_dependencies) > 1 else missing_dependencies[0]
        return error_type(module_name, (object,), {'err': err})

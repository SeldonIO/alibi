"""Functionality for optional importing

This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.
"""


from typing import Union, List, Optional, Any
from string import Template
from importlib import import_module

err_msg_template = Template((
    "Attempted to use $object_name without the correct optional dependencies installed. This may be due to missing"
    + " or incompatible versions of dependencies. To install the correct optional dependencies, run "
    + "`pip install alibi[$missing_dependency]` from the command line. For more information, check the installation"
    + "documentation at https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html."
))


"""Mapping used to ensure correct pip install message is generated if a missing optional dependency is detected. This
dict is used to control two behaviours:

1. When we import objects from missing dependencies we check that any `ModuleNotFoundError` or `ImportError`
corresponds to a missing optional dependency by checking the name of the missing dependency is in `ERROR_TYPES`. We
then map this name to the corresponding optional dependency bucket that will resolve the issue.

2. Some optional dependencies have multiple names such as `torch` and `pytorch`, instead of enforcing a single
naming convention across the whole code base we instead use `ERROR_TYPES` to capture both cases. This is done right
before the pip install message is issued as this is the most robust place to capture these differences.
"""
ERROR_TYPES = {
    'ray': 'ray',
    'tensorflow': 'tensorflow',
    'torch': 'torch',
    'pytorch': 'torch',
    'shap': 'shap',
    'numba': 'shap'
}


class MissingDependency:
    """Missing Dependency Class

    Used to replace any object that requires unmet optional dependencies. Attribute access or calling the __call__
    method on this object will raise an error.
    """
    def __init__(self,
                 object_name: str,
                 err: Union[ModuleNotFoundError, ImportError],
                 missing_dependency: str = 'all',):
        """ Metaclass for MissingDependency classes

        Parameters
        ----------
        object_name
            Name of object we are replacing
        missing_dependency
            Name of missing dependency required for object
        err
            Error to be raised when the class is initialized or used
        """
        self.missing_dependency = missing_dependency
        self.object_name = object_name
        self.err = err

    @property
    def err_msg(self):
        """Generate error message informing user to install missing dependencies."""
        return err_msg_template.substitute(
            object_name=self.object_name,
            missing_dependency=self.missing_dependency)

    def __getattr__(self, key):
        """Raise an error when attributes are accessed."""
        raise ImportError(self.err_msg) from self.err

    def __call__(self, *args, **kwargs):
        """If called, raise an error."""
        raise ImportError(self.err_msg) from self.err


def import_optional(module_name: str, names: Optional[List[str]] = None) -> Any:
    """Import a module that depends on optional dependencies

    Note: This function is used to import modules that depend on optional dependencies. Because it mirrors the python
    import functionality its return type has to be `Any`. Using objects imported with this function can lead to
    misspecification of types as `Any` when the developer intended to be more restrictive.

    Parameters
    ----------
    module_name
        The module to import
    names
        The names to import from the module. If None, all names are imported.

    Returns
    -------
    The module or named objects within the modules if names is not None. If the import fails due to a
    ModuleNotFoundError or ImportError then the requested module or named objects are replaced with instances of
    the MissingDependency class above.
    """

    try:
        module = import_module(module_name)
        # TODO: We should check against specific dependency versions here.
        if names is not None:
            objs = tuple(getattr(module, name) for name in names)
            return objs if len(objs) > 1 else objs[0]
        return module
    except (ImportError, ModuleNotFoundError) as err:
        if err.name is None:
            raise err
        name, *_ = err.name.split('.')
        if name not in ERROR_TYPES:
            raise err
        missing_dependency = ERROR_TYPES[name]
        if names is not None:
            missing_dependencies = \
                tuple(MissingDependency(
                    missing_dependency=missing_dependency,
                    object_name=name,
                    err=err) for name in names)
            return missing_dependencies if len(missing_dependencies) > 1 else missing_dependencies[0]
        return MissingDependency(
            missing_dependency=missing_dependency,
            object_name=module_name,
            err=err)

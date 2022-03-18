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

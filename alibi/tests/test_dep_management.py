"""Test optional dependencies.

These tests import all the named objects from the public API of alibi and test that they throw the correct errors if the
relevant optional dependencies are not installed. If these tests fail, it is likely that:

1. The optional dependency relation hasn't been added to the test script. In this case, this test assumes that your
functionality should work for the default alibi install. If this is not the case you should add the exported object
name to the dependency_map in the relevant test.
2. You haven't protected the relevant export in the public API with the NotInstalled class. In this case, see the docs
string for the utils.notinstalled module.

Notes:
    1. These tests will be skipped in the normal test suite. To run correctly use tox.
    2. If you need to configure a new optional dependency you will need to update the setup.cfg file and add a testenv
    environment as well as to the `extra-requires.txt` file
    3. NBackend functionality may be unique to specific explainers/functions and so there may be multiple such modules
    that need to be tested separately.
"""

from types import ModuleType
from collections import defaultdict
import pytest


def check_correct_dependencies(
        module: ModuleType,
        dependencies: defaultdict,
        opt_dep: str):
    """Checks that imported modules that depend on optional dependencies throw correct errors on use.

    Parameters:
    __________
    module:
        The module to check. Each of the public objects within this module will be checked.
    dependencies:
        A dictionary mapping the name of the object to the list of optional dependencies that it depends on. If the
        key is not in the dictionary, the object is assumed to be independent of optional dependencies. Therefor it
        should pass for the basic alibi install.
    opt_dep:
        The name of the optional dependency that is being tested. This is passed in to this test in the pytest command
        called from the tox env created in the setup.cfg file.
    """
    lib_obj = [obj for obj in dir(module) if not obj.startswith('_')]
    for item in lib_obj:
        item = getattr(module, item)
        if not isinstance(item, ModuleType) and hasattr(item, '__name__'):
            pass_contexts = dependencies[item.__name__]  # type: ignore
            if opt_dep in pass_contexts or 'default' in pass_contexts or opt_dep == 'all':
                with pytest.raises(AttributeError):
                    item.test  # type: ignore # noqa
            else:
                with pytest.raises(ImportError):
                    item.test  # type: ignore # noqa
                # assert('pip install alibi[]' in err.exception)


def test_explainer_dependencies(opt_dep):
    """Tests that the explainer module correctly protects against uninstalled optional dependencies.

    Note: counterfactual rl base and tabular requre one of torch or tensorflow. They will fail if a user tries to
    initialize them without one of tf or torch, but they should still import correctly for all the default install
    option. The backend optional dependency behaviour is tested for in the backend tests and requirement for one of
    torch or tensorflow should be tested in the counterfactual tests themselves.
    """

    explainer_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("AnchorImage", ['default']),
            ("AnchorTabular", ['default']),
            ("AnchorText", ['default']),
            ("CEM", ['tensorflow']),
            ("CounterFactual", ['tensorflow']),
            ("CounterFactualProto", ['tensorflow']),
            ("Counterfactual", ['tensorflow']),
            ("CounterfactualProto", ['tensorflow']),
            ("CounterfactualRL", ['default']),  # See above note
            ("CounterfactualRLTabular", ['default']),  # See above note
            ("DistributedAnchorTabular", ['ray']),
            ("IntegratedGradients", ['tensorflow']),
            ("KernelShap", ['shap']),
            ("TreeShap", ['shap']),
            ("plot_ale", ['default'])]:
        explainer_dependency_map[dependency] = relations
    from alibi import explainers
    check_correct_dependencies(explainers, explainer_dependency_map, opt_dep)


def test_util_dependencies(opt_dep):
    """Tests that the utils module correctly protects against uninstalled optional dependencies."""
    util_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("DistributedExplainer", ['ray']),
            ("LanguageModel", ['tensorflow'])
            ]:
        util_dependency_map[dependency] = relations
    from alibi import utils
    check_correct_dependencies(utils, util_dependency_map, opt_dep)


def test_dataset_dependencies(opt_dep):
    """Tests that the datasets module correctly protects against uninstalled optional dependencies."""
    datasets_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("fetch_fashion_mnist", ['tensorflow'])
            ]:
        datasets_dependency_map[dependency] = relations
    from alibi import datasets
    check_correct_dependencies(datasets, datasets_dependency_map, opt_dep)


def test_confidence_dependencies(opt_dep):
    """Tests that the confidence module correctly protects against uninstalled optional dependencies."""
    confidence_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        confidence_dependency_map[dependency] = relations
    from alibi import confidence
    check_correct_dependencies(confidence, confidence_dependency_map, opt_dep)


def test_cfrl_backend_dependencies(opt_dep):
    """Tests that the backend module correctly protects against uninstalled optional dependencies."""
    backend_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ('alibi.explainers.backends.pytorch.cfrl_base', ['torch']),
        ('alibi.explainers.backends.pytorch.cfrl_tabular', ['torch']),
        ('alibi.explainers.backends.tensorflow.cfrl_base', ['tensorflow']),
        ('alibi.explainers.backends.tensorflow.cfrl_tabular', ['tensorflow']),
    ]:
        backend_dependency_map[dependency] = relations
    from alibi.explainers import backends
    check_correct_dependencies(backends, backend_dependency_map, opt_dep)

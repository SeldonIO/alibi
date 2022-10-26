"""Test optional dependencies.

These tests import all the named objects from the public API of alibi and test that they throw the correct errors if the
relevant optional dependencies are not installed. If these tests fail, it is likely that:

1. The optional dependency relation hasn't been added to the test script. In this case, this test assumes that the
functionality should work for the default alibi install. If this is not the case the exported object name should be
added to the dependency_map in the relevant test.
2. The relevant export in the public API hasn't been imported using `optional_import` from
`alibi.utils.missing_optional_dependency`.

Notes:
    1. These tests will be skipped in the normal test suite. To run correctly use tox.
    2. If you need to configure a new optional dependency you will need to update the setup.cfg file and add a testenv
    environment as well as to the `extra-requires.txt` file
    3. Backend functionality may be unique to specific explainers/functions and so there may be multiple such modules
    that need to be tested separately.
"""

from types import ModuleType
from collections import defaultdict


def check_correct_dependencies(
        module: ModuleType,
        dependencies: defaultdict,
        opt_dep: str):
    """Checks that imported modules that depend on optional dependencies throw correct errors on use.

    Parameters
    ----------
    module
        The module to check. Each of the public objects within this module will be checked.
    dependencies
        A dictionary mapping the name of the object to the list of optional dependencies that it depends on. If a name
        is not in the dictionary, the named object is assumed to be independent of optional dependencies. Therefor it
        should pass for the default alibi install.
    opt_dep
        The name of the optional dependency that is being tested.
    """
    lib_obj = [obj for obj in dir(module) if not obj.startswith('_')]
    for item_name in lib_obj:
        item = getattr(module, item_name)
        if not isinstance(item, ModuleType):
            pass_contexts = dependencies[item_name]  # type: ignore
            try:
                item.test  # type: ignore # noqa
            except AttributeError:
                assert opt_dep in pass_contexts or 'default' in pass_contexts or opt_dep == 'all', \
                    (f'{item_name} was imported instead of an instance of MissingDependency. '
                     f'Are your sure {item} is dependent on {opt_dep}?')
            except ImportError:
                assert opt_dep not in pass_contexts and 'default' not in pass_contexts and opt_dep != 'all', \
                    (f'{item_name} has been imported as an instance of MissingDependency. '
                     f'Are you sure the dependency buckets, {pass_contexts} are correct?')


def test_explainer_dependencies(opt_dep):
    """Tests that the explainer module correctly protects against uninstalled optional dependencies.

    Note: counterfactual rl base and tabular requre one of torch or tensorflow. They will fail if a user tries to
    initialize them without one of tf or torch, but they should still import correctly for the default install option.
    The backend optional dependency behaviour is tested for in the backend tests.
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
            ("LanguageModel", ['tensorflow']),
            ('DistilbertBaseUncased', ['tensorflow']),
            ('BertBaseUncased', ['tensorflow']),
            ('RobertaBase', ['tensorflow'])]:
        util_dependency_map[dependency] = relations
    from alibi import utils
    check_correct_dependencies(utils, util_dependency_map, opt_dep)


def test_dataset_dependencies(opt_dep):
    """Tests that the datasets module correctly protects against uninstalled optional dependencies."""
    datasets_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("fetch_fashion_mnist", ['tensorflow'])]:
        datasets_dependency_map[dependency] = relations
    from alibi import datasets
    check_correct_dependencies(datasets, datasets_dependency_map, opt_dep)


def test_confidence_dependencies(opt_dep):
    """Tests that the confidence module correctly protects against uninstalled optional dependencies."""
    from alibi import confidence
    check_correct_dependencies(confidence, defaultdict(lambda: ['default']), opt_dep)


def test_tensorflow_model_dependencies(opt_dep):
    """Tests that the tensorflow model module correctly protects against uninstalled optional dependencies."""
    tf_model_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ('ADULTEncoder', ['tensorflow']),
            ('ADULTDecoder', ['tensorflow']),
            ('MNISTEncoder', ['tensorflow']),
            ('MNISTDecoder', ['tensorflow']),
            ('MNISTClassifier', ['tensorflow']),
            ('HeAE', ['tensorflow']),
            ('AE', ['tensorflow']),
            ('Actor', ['tensorflow']),
            ('Critic', ['tensorflow'])]:
        tf_model_dependency_map[dependency] = relations
    from alibi.models import tensorflow
    check_correct_dependencies(tensorflow, tf_model_dependency_map, opt_dep)


def test_pytorch_model_dependencies(opt_dep):
    """Tests that the pytorch model module correctly protects against uninstalled optional dependencies."""
    torch_model_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ('ADULTEncoder', ['torch']),
            ('ADULTDecoder', ['torch']),
            ('MNISTEncoder', ['torch']),
            ('MNISTDecoder', ['torch']),
            ('MNISTClassifier', ['torch']),
            ('HeAE', ['torch']),
            ('AE', ['torch']),
            ('Actor', ['torch']),
            ('Critic', ['torch'])]:
        torch_model_dependency_map[dependency] = relations
    from alibi.models import pytorch
    check_correct_dependencies(pytorch, torch_model_dependency_map, opt_dep)

from types import ModuleType
from collections import defaultdict
import pytest


def check_correct_dependencies(module, dependencies, opt_dep):
    lib_obj = [obj for obj in dir(module) if not obj.startswith('_')]
    for item in lib_obj:
        item = getattr(module, item)
        if not isinstance(item, ModuleType) and hasattr(item, '__name__'):
            pass_contexts = dependencies[item.__name__]
            if opt_dep in pass_contexts or 'default' in pass_contexts or opt_dep == 'all':
                assert item.__name__
            else:
                with pytest.raises(ImportError) as err:
                    assert item.__name__
                # assert('pip install alibi[]' in err.exception)
            print(item.__name__, pass_contexts, opt_dep)


def test_explainer_dependencies(opt_dep):
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
            ("CounterfactualRL", ['tensorflow', 'pytorch']),
            ("CounterfactualRLTabular", ['tensorflow', 'pytorch']),
            ("DistributedAnchorTabular", ['ray']),
            ("IntegratedGradients", ['tensorflow']),
            ("KernelShap", ['shap']),
            ("TreeShap", ['shap']),
            ("plot_ale", ['matplotlib'])]:
        explainer_dependency_map[dependency] = relations
    from alibi import explainers
    check_correct_dependencies(explainers, explainer_dependency_map, opt_dep)


def test_util_dependencies(opt_dep):
    util_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        util_dependency_map[dependency] = relations
    from alibi import utils
    check_correct_dependencies(utils, util_dependency_map, opt_dep)


def test_dataset_dependencies(opt_dep):
    datasets_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        datasets_dependency_map[dependency] = relations
    from alibi import datasets
    check_correct_dependencies(datasets, datasets_dependency_map, opt_dep)


def test_confidence_dependencies(opt_dep):
    confidence_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        confidence_dependency_map[dependency] = relations
    from alibi import datasets
    check_correct_dependencies(datasets, confidence_dependency_map, opt_dep)


def test_cfrl_backend_dependencies(opt_dep):
    backend_dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        backend_dependency_map[dependency] = relations
    from alibi.explainers import backends
    check_correct_dependencies(backends, backend_dependency_map, opt_dep)

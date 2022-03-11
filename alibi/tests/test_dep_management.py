from types import ModuleType
from collections import defaultdict
import pytest


dependence_relations = defaultdict(lambda: ['default'])
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
    dependence_relations[dependency] = relations


def test_explainer_dependencies(opt_dep):
    from alibi import explainers
    lib_obj = [obj for obj in dir(explainers) if not obj.startswith('_')]

    for item in lib_obj:
        item = getattr(explainers, item)
        if not isinstance(item, ModuleType):
            pass_contexts = dependence_relations[item.__name__]
            if opt_dep in pass_contexts or 'default' in pass_contexts or opt_dep == 'all':
                item.__name__
            else:
                with pytest.raises(ImportError) as err:
                    item.__name__
                # assert('pip install alibi[]' in err.exception)
            print(item.__name__, pass_contexts, opt_dep)

    assert 1 == 0


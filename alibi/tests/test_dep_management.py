"""
note: https://tox.wiki/en/latest/example/basic.html#passing-down-environment-variables
"""

from types import ModuleType
from collections import defaultdict
import re


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


def check_optional_dependencies():
    deps = defaultdict(set)
    with open('../../extra-requirements.txt') as fp:
        for k in fp:
            if k.startswith('#') or k.startswith('\n'):
                continue
            optional_dependency, bucket_name = k.split(':')
            bucket_name = bucket_name.strip()
            dependency_name = re.split('[<=>]', optional_dependency)[0]
            deps[bucket_name].add(dependency_name)

    buckets = {bucket: True for bucket in deps}

    for bucket in deps:
        for optional_dependency in deps[bucket]:
            try:
                __import__(optional_dependency)
            except ImportError:
                buckets[bucket] = False
                print(optional_dependency, 'is not installed')
    return buckets


def test_explainer_dependencies():
    installed_bucket = check_optional_dependencies()
    print(installed_bucket)

    from alibi import explainers
    lib_obj = [obj for obj in dir(explainers) if not obj.startswith('_')]

    for item in lib_obj:
        item = getattr(explainers, item)
        if not isinstance(item, ModuleType):
            pass_contexts = dependence_relations[item.__name__]
            print(item.__name__, pass_contexts)


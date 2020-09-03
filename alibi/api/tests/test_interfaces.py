import copy
import numpy as np
import pytest
from alibi.api.interfaces import Explainer, Explanation, FitMixin

valid_meta = {"type": "blackbox", "explanations": ['local'], "params": {}}  # type: dict
valid_data = {"anchor": [], "precision": [], "coverage": []}  # type: dict

invalid_meta = []  # type: list
invalid_data = {}  # type: dict

# testing dictionaries
data_dicts = [
    {'b': 2, 'c': 3, 'd': 4},
]


class IncompleteExplainer(Explainer):
    pass


class SimpleExplainer(Explainer):
    def explain(self, X: np.ndarray):
        pass


class SimpleExplainerWithInit(Explainer):

    def __init__(self):
        super().__init__()
        self.meta['params']['a'] = 1

    def explain(self, X: np.ndarray):
        pass


class IncompleteFitExplainer(FitMixin, Explainer):
    def explain(self, X: np.ndarray):
        pass


class SimpleFitExplainer(FitMixin, Explainer):
    def fit(self, X: np.ndarray):
        pass

    def explain(self, X: np.ndarray):
        pass


def test_incomplete_explainer():
    with pytest.raises(TypeError):
        _ = IncompleteExplainer()


def test_explainer():
    exp = SimpleExplainer()
    assert exp.meta["name"] == exp.__class__.__name__
    assert hasattr(exp, "explain")


def test_explainer_with_init():
    exp = SimpleExplainerWithInit()
    assert exp.meta['name'] == exp.__class__.__name__
    assert exp.meta['params'] == {'a': 1}


def test_explainer_valid_meta():
    exp = SimpleExplainer()
    assert hasattr(exp, "explain")


def test_incomplete_fitexplainer():
    with pytest.raises(TypeError):
        _ = IncompleteFitExplainer()


def test_fitexplainer():
    exp = SimpleFitExplainer()
    assert hasattr(exp, "fit")
    assert hasattr(exp, "explain")


def test_explanation():
    exp = Explanation(meta=valid_meta, data=valid_data)
    assert exp.meta == valid_meta
    assert exp.data == valid_data
    assert isinstance(exp, Explanation)

    # test that a warning is raised if accessing attributes as dict keys
    with pytest.warns(None) as record:
        _ = exp['anchor']
    assert len(record) == 1


def test_serialize_deserialize_explanation():
    exp = Explanation(meta=valid_meta, data=valid_data)
    jrep = exp.to_json()
    exp2 = Explanation.from_json(jrep)
    assert exp == exp2


@pytest.mark.parametrize('data_dict', data_dicts)
def test__update_metadata(data_dict):
    exp = SimpleExplainerWithInit()
    meta_init = copy.deepcopy(exp.meta)
    exp._update_metadata(data_dict)
    assert exp.meta == {**meta_init, **data_dict}


@pytest.mark.parametrize('data_dict', data_dicts)
def test__update_metatada_params(data_dict):
    exp = SimpleExplainerWithInit()
    params_init = copy.deepcopy(exp.meta['params'])
    exp._update_metadata(data_dict, params=True)

    assert exp.meta['params'] == {**params_init, **data_dict}

from alibi.explainers.base import Explainer, Explanation, FitMixin
import numpy as np
import pytest
from typing import Dict

meta = {"scope": "local", "type": "blackbox"}
expobj = {"local": {0: None}, "global": None}


class IncompleteExplainer(Explainer):

    @property
    def meta(self):
        return meta

    pass


class SimpleExplainer(Explainer):

    @property
    def meta(self):
        return

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class SimpleExplainerWithMeta(Explainer):

    @property
    def meta(self):
        return meta

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class IncompleteFitExplainer(FitMixin, Explainer):
    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class SimpleFitExplainer(FitMixin, Explainer):

    @property
    def meta(self):
        return

    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        pass

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class IncompleteExplanation(Explanation):

    def __init__(self, meta, data):
        self._meta = meta
        self._data = data

    @property
    def meta(self):
        return self._meta

@pytest.fixture
def min_exp():
    exp = Explanation()
    exp.meta = meta
    exp.data = expobj
    return exp

class CompleteExplanation(Explanation):

    def __init__(self, meta, data):
        self.meta = meta
        self.data = data


def test_incomplete_explainer():
    with pytest.raises(TypeError):
        _ = IncompleteExplainer()


def test_explainer():
    try:
        exp = SimpleExplainer()
        assert exp.meta is None
        assert isinstance(exp.__class__.meta, property)
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_explainer_meta():
    try:
        exp = SimpleExplainerWithMeta()
        assert exp.meta == meta
        assert isinstance(exp.__class__.meta, property)
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_incomplete_fitexplainer():
    with pytest.raises(TypeError):
        _ = IncompleteFitExplainer()


def test_fitexplainer():
    try:
        exp = SimpleFitExplainer()
        assert exp.meta is None
        assert isinstance(exp.__class__.meta, property)
        assert hasattr(exp, "fit")
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_complete_explanation():
    try:
        exp = CompleteExplanation(meta=meta, data=expobj)
        assert exp.meta == meta
        assert isinstance(exp.__class__.meta, property)

        assert exp.data == expobj
        assert exp.data["local"][0] is None
        assert exp.data["global"] is None

        with pytest.raises((IndexError, KeyError)):  # both?
            exp.data["local"][1]

    except Exception:
        pytest.fail("Unknown exception")

def test_minimal_explanation(min_exp):
    try:
        assert min_exp.meta == meta
        assert isinstance(min_exp.__class__.meta, property)

        assert min_exp.data == expobj
        assert min_exp.data["local"][0] is None
        assert min_exp.data["global"] is None

        with pytest.raises((IndexError, KeyError)):  # both?
            min_exp.data["local"][1]

    except Exception:
        pytest.fail("Unknown exception")
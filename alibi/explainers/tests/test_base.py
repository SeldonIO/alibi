from alibi.explainers.base import Explainer, Explanation, FitMixin
import numpy as np
import pytest

meta = {"scope": "local", "type": "blackbox"}


class IncompleteExplainer(Explainer):
    meta = meta
    pass


class SimpleExplainer(Explainer):
    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class SimpleExplainerWithMeta(Explainer):
    meta = meta

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class IncompleteFitExplainer(FitMixin, Explainer):
    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class SimpleFitExplainer(FitMixin, Explainer):
    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        pass

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class CompleteExplanation(Explanation):
    meta = meta
    _exp = {"aggregate": None, "specific": [None]}

    def data(self, key: int = None):
        if key is None:
            return self._exp
        return self._exp["specific"][key]


def test_incomplete_explainer():
    with pytest.raises(TypeError):
        _ = IncompleteExplainer()


def test_explainer():
    try:
        exp = SimpleExplainer()
        assert exp.meta is None
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_explainer_meta():
    try:
        exp = SimpleExplainerWithMeta()
        assert exp.meta == meta
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
        assert hasattr(exp, "fit")
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_complete_explanation():
    try:
        exp = CompleteExplanation()
        assert exp.meta == meta
        assert exp.data() == exp._exp
        assert exp.data(0) is None

        with pytest.raises(IndexError):
            exp.data(1)
    except Exception:
        pytest.fail("Unknown exception")

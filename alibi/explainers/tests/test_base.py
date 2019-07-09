from alibi.explainers.base import Explainer, Explanation, FitMixin
import numpy as np
import pytest


class EmptyExplainer(Explainer):
    pass


class IncompleteExplainer(Explainer):
    explainer_type = "local"
    pass


class SimpleExplainer(Explainer):
    explainer_type = "local"

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class IncompleteFitExplainer(FitMixin, Explainer):
    explainer_type = "local"

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class SimpleFitExplainer(FitMixin, Explainer):
    explainer_type = "local"

    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        pass

    def explain(self, X: np.ndarray, y: np.ndarray = None):
        pass


class CompleteExplanation(Explanation):
    explanation_type = "local"
    _exp = {"aggregate": None, "specific": [None]}

    def data(self, key: int = None):
        if key is None:
            return self._exp
        return self._exp["specific"][key]


def test_empty_explainer():
    with pytest.raises(TypeError):
        _ = EmptyExplainer()


def test_incomplete_explainer():
    with pytest.raises(TypeError):
        _ = IncompleteExplainer()


def test_explainer():
    try:
        exp = SimpleExplainer()
        assert exp.explainer_type == "local"

    except Exception:
        pytest.fail("Unknown exception")


def test_incomplete_fitexplainer():
    with pytest.raises(TypeError):
        _ = IncompleteFitExplainer()


def test_fitexplainer():
    try:
        exp = SimpleFitExplainer()
        assert exp.explainer_type == "local"
    except Exception:
        pytest.fail("Unknown exception")


def test_complete_explanation():
    try:
        exp = CompleteExplanation()
        assert exp.explanation_type == "local"
        assert exp.data() == exp._exp
        assert exp.data(0) is None

        with pytest.raises(IndexError):
            exp.data(1)
    except Exception:
        pytest.fail("Unknown exception")

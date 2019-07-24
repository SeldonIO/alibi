from alibi.explainers.base import Explainer, Explanation, FitMixin, DataException, MetaException
import numpy as np
import pytest

valid_meta = {"scope": "local", "type": "blackbox"}
valid_data = {"local": {0: None}, "global": None}

invalid_meta = []
invalid_data = {}


class IncompleteExplainer(Explainer):
    pass


class SimpleExplainer(Explainer):

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


@pytest.fixture
def min_exp():
    exp = Explanation()
    exp.meta = valid_meta
    exp.data = valid_data
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
        assert exp.meta == {}
        assert isinstance(exp.__class__.meta, property)
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_explainer_valid_meta():
    try:
        exp = SimpleExplainer()
        exp.meta = valid_meta
        assert exp.meta == valid_meta
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
        assert exp.meta == {}
        assert isinstance(exp.__class__.meta, property)
        assert hasattr(exp, "fit")
        assert hasattr(exp, "explain")
    except Exception:
        pytest.fail("Unknown exception")


def test_complete_explanation():
    try:
        exp = CompleteExplanation(meta=valid_meta, data=valid_data)
        assert exp.meta == valid_meta
        assert isinstance(exp.__class__.meta, property)

        assert exp.data == valid_data
        assert exp.data["local"][0] is None
        assert exp.data["global"] is None

        with pytest.raises((IndexError, KeyError)):  # both?
            exp.data["local"][1]

    except Exception:
        pytest.fail("Unknown exception")


def test_minimal_explanation(min_exp):
    try:
        assert min_exp.meta == valid_meta
        assert isinstance(min_exp.__class__.meta, property)

        assert min_exp.data == valid_data
        assert min_exp.data["local"][0] is None
        assert min_exp.data["global"] is None

        with pytest.raises((IndexError, KeyError)):  # both?
            min_exp.data["local"][1]

    except Exception:
        pytest.fail("Unknown exception")


def test_invalid_explanation():
    with pytest.raises(DataException):
        exp = CompleteExplanation(meta=valid_meta, data=invalid_data)
    with pytest.raises(MetaException):
        exp = CompleteExplanation(meta=invalid_meta, data=valid_data)
    with pytest.raises(MetaException):
        exp = CompleteExplanation(meta=invalid_meta, data=invalid_data)

from alibi.explainers.base import BaseExplainer, BaseExplanation, FitMixin, DataException, MetaException
import numpy as np
import pytest

valid_meta = {"scope": "local", "type": "blackbox"}
valid_data = {"local": [None], "overall": None}

invalid_meta = []
invalid_data = {}


class IncompleteExplainer(BaseExplainer):
    pass


class SimpleExplainer(BaseExplainer):

    def explain(self, X: np.ndarray):
        pass


class IncompleteFitExplainer(FitMixin, BaseExplainer):

    def explain(self, X: np.ndarray):
        pass


class SimpleFitExplainer(FitMixin, BaseExplainer):

    def fit(self, X: np.ndarray):
        pass

    def explain(self, X: np.ndarray):
        pass


@pytest.fixture
def min_exp():
    exp = BaseExplanation()
    exp.meta = valid_meta
    exp.data = valid_data
    return exp


class CompleteExplanation(BaseExplanation):

    def __init__(self, meta, data):
        self.meta = meta
        self.data = data


def test_incomplete_explainer():
    with pytest.raises(TypeError):
        _ = IncompleteExplainer()


def test_explainer():
    try:
        exp = SimpleExplainer()
        assert exp.meta["name"] == exp.__class__.__name__
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
        assert exp.meta["name"] == exp.__class__.__name__
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
        assert exp.data["overall"] is None

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
        assert min_exp.data["overall"] is None

        with pytest.raises((IndexError, KeyError)):  # both?
            min_exp.data["local"][1]

    except Exception:
        pytest.fail("Unknown exception")


def test_invalid_explanation():
    with pytest.raises(DataException):
        CompleteExplanation(meta=valid_meta, data=invalid_data)
    with pytest.raises(MetaException):
        CompleteExplanation(meta=invalid_meta, data=valid_data)
    with pytest.raises(MetaException):
        CompleteExplanation(meta=invalid_meta, data=invalid_data)

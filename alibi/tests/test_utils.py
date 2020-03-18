import pytest

import numpy as np

from alibi.tests.utils import MockPredictor
from alibi.utils.wrappers import ArgmaxTransformer

out_dim = [1, 5]      # number of classes
out_type = ['proba']  # classifier returns probabilities


@pytest.mark.parametrize("out_dim", out_dim, ids="out_dim={}".format)
@pytest.mark.parametrize("out_type", out_type, ids="out_type={}".format)
def test_argmax_transformer(monkeypatch, out_dim, out_type):
    """
    Test that the conversion of output probabilities to class labels works
    for a range of classifier output types.
    """

    # setup a transformer with a mock predictor
    transformer = ArgmaxTransformer(None)
    monkeypatch.setattr(
        transformer,
        "predictor",
        MockPredictor(
            out_dim,
            out_type=out_type,
        )
    )
    # fake predictor input
    X = np.random.random(size=(50, 14))

    result = transformer(X)
    # argmax transformer should do get rid of the feature dimension
    assert len(result.shape) == len(X.shape) - 1

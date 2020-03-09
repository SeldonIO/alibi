import pytest

import numpy as np

from itertools import product

from alibi.tests.utils import MockPredictor
from alibi.utils.wrappers import ArgmaxTransformer


#####################################################################
X = np.random.randint(0, 10, size=5)  # Fake classifier input
out_dim = [1, 5]  # number of classes
out_type = ['proba']
sz = [None, 5]    # mock classifier returns a tensor of [sz, out_dim]
predictor_settings = list(product(out_dim, out_type, sz))
#####################################################################


# parametrize a classifier with different output dimensions and output types,
# defined in the test setup above.
@pytest.fixture(params=predictor_settings, ids='predictor_settings={}'.format)
def patch_transformer(monkeypatch, request):
    out_dim, out_type, sz = request.param
    transformer = ArgmaxTransformer(None)
    monkeypatch.setattr(
        transformer,
        "predictor",
        MockPredictor(
            out_dim,
            out_type=out_type,
            sz=sz,
            key=(out_dim, out_type, sz)
        )
    )

    return transformer


# Test that the conversion of output probabilities to class labels works
# for a range of classifier output types
def test_argmax_transformer(monkeypatch, patch_transformer):

    transformer = patch_transformer

    result = transformer(X)
    if transformer.predictor.sz is not None:
        # batch prediction
        assert result.size == transformer.predictor.sz
    else:
        # batch size = 1
        assert result.size == 1

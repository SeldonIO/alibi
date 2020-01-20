import pytest

import numpy as np

from itertools import product
from alibi.utils.wrappers import ArgmaxTransformer
from typing import Union, Tuple

OUT_TYPES = ['proba', 'class']

#####################################################################
X = np.random.randint(0, 10, size=5)  # Fake classifier input
out_dim = [1, 5]  # number of classes
out_type = ['proba']
sz = [None, 5]    # mock classifier returns a tensor of [sz, out_dim]
predictor_settings = list(product(out_dim, out_type, sz))
#####################################################################


def id_func(param):
    """
    Shows the names of the parameters in test names
    """
    return repr(param)


class MockPredictor:
    """
    A class the mimicks the output of a classifier to
    allow testing of functionality that depends on it without
    inference overhead.
    """
    def __init__(self, out_dim: int, out_type='proba', sz: Union[Tuple, int] = None, key: tuple = None) -> None:
        """
        Parameters
        ----------
            out_dim
                The number of output classes.
            out_type
                Indicates if probabilities or class predictions are generated.
            sz
                If out_type is proba, tensor of size [sz, out_dim] is returned if
                    sz is int, and of size [*size, out_dim] otherwise.
                If out_type is class, then a tensor of size sz is returned.
        """
        self.out_dim = out_dim
        self.out_type = out_type
        self.sz = sz
        self.key = key
        if out_type not in OUT_TYPES:
            raise ValueError("Unknown output type. Accepted values are {}".format(OUT_TYPES))

    def __call__(self, *args, **kwargs):
        # can specify size s.t. multiple predictions/batches of predictions are returned

        if self.out_type == 'proba':
            return self._generate_probas(self.sz, *args, **kwargs)
        else:
            return self._generate_labels(self.sz, *args, **kwargs)

    def _generate_probas(self, sz: tuple = None, *args, **kwargs) -> np.ndarray:
        """
        Generates probability vectors by sampling from a Dirichlet distribution.
        User can specify the Dirichlet distribution parameters via the 'alpha'
        kwargs. See documentation for np.random.dirichlet  to see how to set
        this parameter.

        Parameters
        ----------
        sz
            Output dimension: [N, B] where N is number of batches and B is batch size.
        """

        # set distribution parameters
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
            if isinstance(alpha, np.ndarray):
                (dim,) = alpha.squeeze().shape
            elif isinstance(alpha, list):
                dim = len(alpha)
            else:
                raise TypeError("Expected alpha to be of type list or np.ndarray!")

            try:
                assert dim == self.out_dim
            except AssertionError:
                raise ValueError("The dimension of the Dirichlet distribution parameters"
                                 "must match output dimension. Got alpha dim={} and "
                                 "out_dim={} ".format(dim, self.out_dim))
        else:
            alpha = np.ones(self.out_dim)

        return np.random.dirichlet(alpha, size=sz)

    def _generate_labels(self, sz: tuple = None, *args, **kwargs) -> np.ndarray:
        """
        Generates labels by sampling random integers in range(0, n_classes+1).
        """
        if sz:
            sz += (self.out_dim,)
        return np.random.randint(0, self.out_dim + 1, size=sz)

# parametrize classifier with different output dimensions and output types,
# defined in test setup
@pytest.fixture(params=predictor_settings, ids=id_func)
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

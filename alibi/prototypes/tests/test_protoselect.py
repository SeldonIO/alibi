import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from alibi.prototypes import ProtoSelect
from alibi.utils.kernel import EuclideanDistance
from alibi.prototypes.protoselect import cv_protoselect_euclidean


@pytest.mark.parametrize('n_classes', [2, 3, 5, 10])
@pytest.mark.parametrize('ft_factor', [1, 2, 3, 10])
@pytest.mark.parametrize('kernel_distance', [EuclideanDistance()])
@pytest.mark.parametrize('num_prototypes', [30, 40, 50, 100])
@pytest.mark.parametrize('eps', [0.2, 0.5])
def test_protoselect(n_classes, ft_factor, kernel_distance, num_prototypes, eps):
    """ ProtoSelect integration test on a multiclass dataset."""
    X, Y = make_classification(n_samples=1000,
                               n_features=ft_factor * n_classes,
                               n_informative=n_classes,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=1,
                               class_sep=10,
                               random_state=0)

    # define & fit the explainer
    explainer = ProtoSelect(eps=eps, kernel_distance=kernel_distance)
    explainer = explainer.fit(X_ref=X, Y_ref=Y)

    # get prototypes
    explanation = explainer.explain(num_prototypes=num_prototypes)
    protos = explanation.prototypes
    protos_indices = explanation.prototypes_indices
    protos_labels = explanation.prototypes_labels

    assert len(protos) == len(protos_indices) == len(protos_labels)
    assert len(protos) <= num_prototypes
    assert set(protos_labels).issubset(set(Y))


@pytest.mark.parametrize('n_classes', [2])
@pytest.mark.parametrize('use_protos', [False, True])
@pytest.mark.parametrize('use_valset', [False, True])
@pytest.mark.parametrize('num_prototypes', [10, 30])
@pytest.mark.parametrize('eps_grid', [None, np.arange(15)])
@pytest.mark.parametrize('quantiles', [(0., 1.), (0.1, 0.9), (0.1, 1.), (0., 0.4)])
@pytest.mark.parametrize('grid_size', [2, 10, 20])
@pytest.mark.parametrize('n_splits', [2, 5])
@pytest.mark.parametrize('batch_size', [100])
def test_cv_protoselect_euclidean(n_classes, use_protos, use_valset, num_prototypes, eps_grid, quantiles, grid_size,
                                  n_splits, batch_size):
    """
    Unit test for cross-validation. Checks if all parameters are passed correctly and checks the
    appropriate behavior when the validation is passed or omitted.
    """
    X, Y = make_classification(n_samples=1000,
                               n_features=n_classes,
                               n_informative=n_classes,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=1,
                               class_sep=10,
                               random_state=0)

    # construct datasets
    if use_valset:
        X_ref, X_val, Y_ref, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
        refset = (X_ref, Y_ref)
        protoset = (X_ref,) if use_protos else None
        valset = (X_val, Y_val)
    else:
        refset = (X, Y)
        protoset = (X,) if use_protos else None
        valset = None

    cv = cv_protoselect_euclidean(refset=refset,
                                  protoset=protoset,
                                  valset=valset,
                                  num_prototypes=num_prototypes,
                                  eps_grid=eps_grid,
                                  quantiles=quantiles,
                                  grid_size=grid_size,
                                  n_splits=n_splits,
                                  batch_size=batch_size)

    if use_valset:
        # check if the `scores` shape is 1D when validation set is passed
        assert len(cv['meta']['scores']) == len(cv['meta']['eps_grid'])
    else:
        # check if the `scores` shape is 2D when no validation is passed
        assert cv['meta']['scores'].shape == (len(cv['meta']['eps_grid']), n_splits)

    if eps_grid is not None:
        # if `eps_grid` is provided, the check that the `best_eps` is amongst `eps_grid` values.
        assert np.any(np.isclose(eps_grid, cv['best_eps']))
    else:
        # if `eps_grid` is not provided, check that the search interval was split in `grid_size` bins
        assert len(cv['meta']['eps_grid']) == grid_size

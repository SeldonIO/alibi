from copy import deepcopy

import matplotlib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from alibi.api.defaults import (DEFAULT_DATA_PROTOSELECT,
                                DEFAULT_META_PROTOSELECT)
from alibi.api.interfaces import Explanation
from alibi.prototypes import ProtoSelect
from alibi.prototypes.protoselect import (_batch_preprocessing, _imscatterplot,
                                          compute_prototype_importances,
                                          cv_protoselect_euclidean,
                                          visualize_image_prototypes)
from alibi.utils.kernel import EuclideanDistance


@pytest.mark.parametrize('n_classes', [2, 3, 5, 10])
@pytest.mark.parametrize('ft_factor', [1, 2, 3, 10])
@pytest.mark.parametrize('kernel_distance', [EuclideanDistance()])
@pytest.mark.parametrize('num_prototypes', [30, 40, 50, 100])
@pytest.mark.parametrize('eps', [0.2, 0.5])
def test_protoselect(n_classes, ft_factor, kernel_distance, num_prototypes, eps):
    """ ProtoSelect integration test on a multiclass dataset."""
    X, y = make_classification(n_samples=1000,
                               n_features=ft_factor * n_classes,
                               n_informative=n_classes,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=1,
                               class_sep=10,
                               random_state=0)

    # define & fit the summariser
    summariser = ProtoSelect(eps=eps, kernel_distance=kernel_distance)
    summariser = summariser.fit(X=X, y=y)

    # get prototypes
    summary = summariser.summarise(num_prototypes=num_prototypes)
    protos = summary.prototypes
    protos_indices = summary.prototype_indices
    protos_labels = summary.prototype_labels

    assert len(protos) == len(protos_indices) == len(protos_labels)
    assert len(protos) <= num_prototypes
    assert set(protos_labels).issubset(set(y))


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
    X_train, y_train = make_classification(n_samples=1000,
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
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
        trainset = (X_train, y_train)
        protoset = (X_train,) if use_protos else None
        valset = (X_val, y_val)
    else:
        trainset = (X_train, y_train)
        protoset = (X_train,) if use_protos else None
        valset = None

    cv = cv_protoselect_euclidean(trainset=trainset,
                                  protoset=protoset,
                                  valset=valset,
                                  num_prototypes=num_prototypes,
                                  eps_grid=eps_grid,
                                  quantiles=quantiles,
                                  grid_size=grid_size,
                                  kfold_kw={'n_splits': n_splits},
                                  protoselect_kw={'batch_size': batch_size})

    if use_valset:
        # check if the `scores` shape is 1D when validation set is passed
        assert cv['meta']['scores'].shape == (len(cv['meta']['eps_grid']), 1)
    else:
        # check if the `scores` shape is 2D when no validation is passed
        assert cv['meta']['scores'].shape == (len(cv['meta']['eps_grid']), n_splits)

    if eps_grid is not None:
        # if `eps_grid` is provided, the check that the `best_eps` is amongst `eps_grid` values.
        assert np.any(np.isclose(eps_grid, cv['best_eps']))
    else:
        # if `eps_grid` is not provided, check that the search interval was split in `grid_size` bins
        assert len(cv['meta']['eps_grid']) == grid_size


@pytest.mark.parametrize('n_samples', [10, 50, 100])
@pytest.mark.parametrize('n_classes', [1000])
def test_relabeling(n_samples, n_classes):
    """
    Tests whether the internal relabeling works properly. For example, if the labels provided were `[40, 51]`,
    internally, we relabel them as `[0, 1]`.
    """

    X, y = make_classification(n_samples=n_samples,
                               n_features=n_classes,
                               n_informative=n_classes,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=1,
                               class_sep=10,
                               random_state=0)

    # define summariser and obtain summary
    summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0.5)
    summariser = summariser.fit(X=X, y=y)
    summary = summariser.summarise(num_prototypes=np.random.randint(1, n_samples, 1).item())

    # check internal Y_ref relabeling
    provided_labels = np.unique(y)
    internal_labels = np.unique(summariser.y)
    assert np.array_equal(internal_labels, np.arange(len(provided_labels)))

    # check if the prototypes labels are labels with the provided labels
    assert np.all(np.isin(np.unique(summary.data['prototype_labels']), provided_labels))


def test_size_match():
    """
    Tests if the error is raised when the number of data instance does not match the number of labels.
    """
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 10, 50)

    summariser = ProtoSelect(eps=0.5, kernel_distance=EuclideanDistance())
    with pytest.raises(ValueError):
        summariser.fit(X=X, y=y)


@pytest.mark.parametrize('batch_size', [2, 8, 16])
def test__batch_preprocessing(batch_size):
    """ Test if the batch preprocessing function returns the correct shapes. """
    n_samples, n_features, n_removed = 50, 5, 2
    X = np.random.randn(n_samples, n_features)

    def preprocess_fn(X):
        return X[:, :-n_removed]

    X_ft = _batch_preprocessing(X=X, preprocess_fn=preprocess_fn, batch_size=batch_size)
    assert X_ft.shape == (n_samples, n_features - n_removed)


@pytest.fixture(scope='module')
def importance_data():
    X = np.array([
        [0.548, 0.715],
        [0.602, 0.544],
        [0.423, 0.645],
        [5.437, 4.891],
        [5.963, 4.383],
        [5.791, 4.528],
        [5.568, 4.925],
        [5.071, 4.087],
        [-3.979, 4.832],
        [-3.221, 4.870],
        [-3.021, 4.799],
        [-3.538, 4.780],
        [-3.881, 4.639],
        [-3.856, 4.944],
        [-3.478, 4.414]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    trainset = (X, y)

    meta = deepcopy(DEFAULT_META_PROTOSELECT)
    data = deepcopy(DEFAULT_DATA_PROTOSELECT)
    meta['params'] = {
        'kernel_distance': 'EuclideanDistance',
        'eps': 0.5,
        'lambda_penalty': 0.066,
        'batch_size': 10000000000,
        'verbose': True
    }
    data['prototypes'] = np.array([
        [0.548, 0.715],
        [5.791, 4.528],
        [-3.53, 4.780]
    ])
    data['prototype_indices'] = np.array([0, 5, 11], dtype=np.int32),
    data['prototype_labels'] = np.array([0, 1, 2], dtype=np.int32)
    summary = Explanation(meta=meta, data=data)
    return trainset, summary


def test_compute_prototype_importances(importance_data):
    """ Test if the computation of the prototype importances returns the expected results for a simple
    example with three clusters and three prototypes. In this case, the importance of each prototype is
    given by the number of data instances in each cluster. """
    trainset, summary = importance_data
    X, y = trainset

    expected_importances = np.unique(y, return_counts=True)[1]
    importances = compute_prototype_importances(summary=summary, trainset=trainset)
    assert np.allclose(expected_importances, importances['prototype_importances'])


@pytest.fixture(scope='module')
def plot_data():
    n_samples = 10
    x = np.random.uniform(low=-10, high=10, size=(n_samples, ))
    y = np.random.uniform(low=-10, high=10, size=(n_samples, ))

    image_size = (5, 5)
    images = np.random.rand(n_samples, *image_size, 3)

    zoom_lb, zoom_ub = 3, 7
    zoom = np.random.permutation(np.linspace(1, 5, n_samples))
    return {
        'x': x,
        'y': y,
        'image_size': image_size,
        'images': images,
        'zoom_lb': zoom_lb,
        'zoom_ub': zoom_ub,
        'zoom': zoom
    }


@pytest.mark.parametrize('use_zoom', [False, True])
def test__imscatterplot(plot_data, use_zoom):
    """ Test `_imscatterplot` function. """
    ax = _imscatterplot(x=plot_data['x'],
                        y=plot_data['y'],
                        images=plot_data['images'],
                        image_size=plot_data['image_size'],
                        zoom=plot_data['zoom'] if use_zoom else None,
                        zoom_lb=plot_data['zoom_lb'],
                        zoom_ub=plot_data['zoom_ub'],
                        sort_by_zoom=True)

    annboxes = [x for x in ax.get_children() if isinstance(x, matplotlib.offsetbox.AnnotationBbox)]
    data = np.array([annbox.offsetbox.get_data() for annbox in annboxes])
    zoom = np.array([annbox.offsetbox.get_zoom() for annbox in annboxes])

    sorted_idx = np.argsort(plot_data['zoom'])[::-1] if use_zoom else None
    expected_data = plot_data['images'][sorted_idx]

    if not use_zoom:
        expected_zoom = np.ones(len(plot_data['zoom']))
    else:
        expected_zoom = plot_data['zoom'][sorted_idx]
        expected_zoom = (expected_zoom - expected_zoom.min()) / (expected_zoom.max() - expected_zoom.min())
        expected_zoom = expected_zoom * (plot_data['zoom_ub'] - plot_data['zoom_lb']) + plot_data['zoom_lb']

    assert np.allclose(expected_data, data)
    assert np.allclose(expected_zoom, zoom)


def test_visualize_image_prototypes(mocker):
    """ Test the `visualize_image_prototypes` function. """
    importances = {
        'prototype_importances': np.random.randint(2, 50, size=(10, )),
        'X_protos': np.random.randn(10, 2),
        'X_protos_ft': None
    }

    m1 = mocker.patch('alibi.prototypes.protoselect.compute_prototype_importances', return_value=importances)
    m2 = mocker.patch('alibi.prototypes.protoselect._imscatterplot')
    visualize_image_prototypes(summary=None, trainset=None, reducer=lambda x: x)
    m1.assert_called_once()
    m2.assert_called_once()

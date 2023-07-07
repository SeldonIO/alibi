import re
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from packaging import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch, shuffle
import sklearn

from alibi.api.defaults import DEFAULT_DATA_PD, DEFAULT_META_PD
from alibi.api.interfaces import Explanation
from alibi.explainers import PartialDependence, TreePartialDependence, plot_pd
from alibi.explainers.partial_dependence import (_plot_one_pd_cat,
                                                 _plot_one_pd_num,
                                                 _plot_two_pd_cat_cat,
                                                 _plot_two_pd_num_cat,
                                                 _plot_two_pd_num_num,
                                                 _process_pd_ice, _sample_ice)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


@pytest.fixture(scope='module')
def binary_data():
    n_samples, n_features, n_informative, n_classes = 200, 100, 30, 2
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_classes=n_classes,
                               random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': None,
    }


@pytest.fixture(scope='module')
def multioutput_dataset():
    n_samples, n_features, n_informative, n_classes = 200, 100, 30, 3
    X, y1 = make_classification(n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                n_classes=n_classes,
                                random_state=0)

    y2 = shuffle(y1, random_state=1)
    y3 = shuffle(y1, random_state=2)
    Y = np.vstack((y1, y2, y3)).T
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'preprocessor': None
    }


@pytest.fixture(scope='module')
def multioutput_classifier(request):
    predictor = request.param
    return MultiOutputClassifier(predictor)


@pytest.mark.parametrize('predictor', [GradientBoostingClassifier()])
def test_unfitted_estimator(predictor):
    """ Checks if raises error for unfitted model. """
    with pytest.raises(NotFittedError) as err:
        TreePartialDependence(predictor=predictor)
    assert re.search('not fitted yet', err.value.args[0])


@pytest.mark.parametrize('multioutput_classifier', [GradientBoostingClassifier()], indirect=True)
def test_multioutput_estimator(multioutput_classifier, multioutput_dataset):
    """ Check if raises error for multi-output model"""
    X_train, Y_train = multioutput_dataset['X_train'], multioutput_dataset['Y_train']
    multioutput_classifier.fit(X_train, Y_train)
    with pytest.raises(ValueError) as err:
        TreePartialDependence(predictor=multioutput_classifier)
    assert re.search('multiclass-multioutput', err.value.args[0].lower())


@pytest.mark.parametrize('kind', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_kind(rf_classifier, kind):
    """ Checks if raises error for unknown `kind`. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor.predict_proba,)
    with pytest.raises(ValueError) as err:
        explainer.explain(X=None, kind=kind)
    assert re.search("``kind=\'\w+\'`` is invalid", err.value.args[0].lower())  # noqa: W605


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unsupported_method_recursion(rf_classifier):
    """ Checks if raises error when a model which does not support method recursion is passed to the
    `TreePartialDependence`. """
    predictor, _ = rf_classifier
    with pytest.raises(ValueError) as err:
        TreePartialDependence(predictor=predictor)
    assert re.search("`TreePartialDependence` only supports by the following estimators:", err.value.args[0])


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('grid_resolution', [3, 5, np.inf])
@pytest.mark.parametrize('features', [
    [0, 1, (0, 1), (1, 2)]
])
def test_explanation_numerical_shapes(rf_classifier, iris_data, grid_resolution, features):
    """ Checks the correct shapes of the arrays contained in the explanation object of numerical features
    for the black-box implementation. """
    predictor, _ = rf_classifier
    X_train, y_train = iris_data['X_train'][:30], iris_data['y_train'][:30]
    num_targets = len(np.unique(y_train))
    num_instances = len(X_train)

    explainer = PartialDependence(predictor=predictor.predict_proba)
    exp = explainer.explain(X=X_train,
                            features=features,
                            grid_resolution=grid_resolution,
                            kind='both')

    # check that the values returned match the number of requested features
    assert len(exp.data['feature_names']) == len(features)
    assert len(exp.data['pd_values']) == len(features)
    assert len(exp.data['ice_values']) == len(features)
    assert len(exp.data['feature_values']) == len(features)

    for i, f in enumerate(features):
        if isinstance(f, Tuple):
            # check deciles
            assert isinstance(exp.data['feature_deciles'][i], List)
            assert len(exp.data['feature_deciles'][i]) == len(f)
            assert len(exp.data['feature_deciles'][i][0]) == 11
            assert len(exp.data['feature_deciles'][i][1]) == 11

            # check feature_values
            assert isinstance(exp.data['feature_values'][i], List)
            assert len(exp.data['feature_values'][i]) == len(f)
            assert len(exp.data['feature_values'][i][0]) == (len(np.unique(X_train[:, f[0]]))
                                                             if grid_resolution == np.inf else grid_resolution)
            assert len(exp.data['feature_values'][i][1]) == (len(np.unique(X_train[:, f[1]]))
                                                             if grid_resolution == np.inf else grid_resolution)

            # check pd_values
            assert exp.data['pd_values'][i].shape == (num_targets,
                                                      len(exp.data['feature_values'][i][0]),
                                                      len(exp.data['feature_values'][i][1]))

            # check ice_values
            assert exp.data['ice_values'][i].shape == (num_targets,
                                                       num_instances,
                                                       len(exp.data['feature_values'][i][0]),
                                                       len(exp.data['feature_values'][i][1]))

        else:
            # check feature_deciles
            assert len(exp.data['feature_deciles'][i]) == 11

            # check feature_values
            assert len(exp.data['feature_values'][i]) == (len(np.unique(X_train[:, f])) if grid_resolution == np.inf
                                                          else grid_resolution)

            # check pd_values
            assert exp.data['pd_values'][i].shape == (num_targets, len(exp.data['feature_values'][i]))

            # check ice_value
            assert exp.data['ice_values'][i].shape == (num_targets, num_instances, len(exp.data['feature_values'][i]))


@pytest.mark.parametrize('rf_regressor', [lazy_fixture('diabetes_data')], indirect=True)
@pytest.mark.parametrize('kind', ['average', 'individual', 'both'])
@pytest.mark.parametrize('features', [
    [0, 1, 2],
    [(0, 1), (0, 2), (1, 2)],
    [0, 1, (0, 1)]
])
def test_blackbox_regression(rf_regressor, diabetes_data, kind, features):
    """ Test the black-box predictor for a regression function. """
    rf, _ = rf_regressor
    X_train = diabetes_data['X_train']

    # define explainer and compute explanation
    explainer = PartialDependence(predictor=rf.predict)
    explainer.explain(X=X_train,
                      features=features,
                      kind=kind,
                      grid_resolution=10)


@pytest.mark.parametrize('lr_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('kind', ['average', 'individual', 'both'])
@pytest.mark.parametrize('features', [
    [0, 1, 2],
    [(0, 1), (0, 2), (1, 2)],
    [0, 1, (0, 1)]
])
def test_blackbox_classification(lr_classifier, iris_data, kind, features):
    """ Test the black-box predictor for a classification function. """
    X_train, _ = iris_data['X_train'], iris_data['y_train']
    lr, _ = lr_classifier

    # define explainer and compute explanation
    explainer = PartialDependence(predictor=lr.predict_proba)
    explainer.explain(X=X_train,
                      features=features,
                      kind=kind,
                      grid_resolution=10)


@pytest.mark.parametrize('use_int', [False, True])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
def test_grid_points(adult_data, rf_classifier, use_int):
    """ Checks whether the grid points provided are used for computing the partial dependencies. """
    rf, preprocessor = rf_classifier
    prediction_fn = lambda x: rf.predict_proba(preprocessor.transform(x))  # noqa E731

    feature_names = adult_data['metadata']['feature_names']
    categorical_names = adult_data['metadata']['category_map']
    X_train = adult_data['X_train']

    # construct random grid_points by choosing random values between min and max for each numerical feature,
    # and sampling at random from the categorical names for each categorical feature.
    grid_points = {}
    for i in range(len(feature_names)):
        if i not in categorical_names:
            min_val, max_val = X_train[:, i].min(), X_train[:, i].max()
            size = np.random.randint(low=1, high=len(np.unique(X_train[:, i])))
            vals = np.random.uniform(min_val, max_val, size=size)
        else:
            size = np.random.randint(low=1, high=len(categorical_names[i]))
            categorical_values = np.arange(len(categorical_names[i])) if use_int else categorical_names[i]
            vals = np.random.choice(categorical_values, size=size, replace=False)

        grid_points[i] = vals

    # define explainer
    explainer = PartialDependence(predictor=prediction_fn,
                                  feature_names=feature_names,
                                  categorical_names=categorical_names)

    # compute explanation for every feature using the grid_points
    exp = explainer.explain(X=X_train[:100],
                            features=None,
                            kind='average',
                            grid_points=grid_points)

    for i in range(len(feature_names)):
        np.testing.assert_allclose(exp.data['feature_values'][i], grid_points[i])


@pytest.mark.parametrize('use_int', [False, True])
def test_grid_points_error(adult_data, use_int):
    """ Checks if the `_grid_points_sanity_checks` throw an error when the `grid_points` for a categorical feature
    are not a subset of the feature values provided in `categorical_names`. """
    feature_names = adult_data['metadata']['feature_names']
    categorical_names = adult_data['metadata']['category_map']
    X_train = adult_data['X_train']

    # construct random `grid_points` by choosing random values between min and max for each numerical feature,
    # and sampling at random from the categorical names for each categorical feature.
    grid_points = {}
    for i in range(len(feature_names)):
        if i in categorical_names:
            size = np.random.randint(low=1, high=len(categorical_names[i]))
            categorical_values = np.arange(len(categorical_names[i])) if use_int else categorical_names[i]
            grid_points[i] = np.random.choice(categorical_values, size=size, replace=False)

            # append a wrong value
            grid_points[i] = np.append(grid_points[i], len(categorical_values) if use_int else '[UNK]')

    # define explainer
    explainer = PartialDependence(predictor=lambda x: np.zeros(x.shape[0]),
                                  feature_names=feature_names,
                                  categorical_names=categorical_names)

    # compute explanation for every feature using the `grid_points`
    with pytest.raises(ValueError):
        explainer._grid_points_sanity_checks(grid_points, n_features=X_train.shape[1])


def assert_feature_values_equal(exp_alibi: Explanation, exp_sklearn: Bunch):
    """ Compares feature values of `alibi` explanation and `sklearn` explanation. """
    if isinstance(exp_alibi.data['feature_names'][0], tuple):
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.data['feature_values'][0][i], exp_sklearn['values'][i])
    else:
        assert np.allclose(exp_alibi.data['feature_values'][0], exp_sklearn['values'][0])


def get_alibi_pd_explanation(predictor: BaseEstimator,
                             X: np.ndarray,
                             features: List[Union[int, Tuple[int, int]]],
                             kind: Literal['average', 'individual', 'both'],
                             percentiles: Tuple[float, float],
                             grid_resolution: int,
                             feature_names: Optional[List[str]] = None,
                             categorical_names: Optional[Dict[int, List[str]]] = None):
    """ Computes `alibi` pd explanation. """
    explainer = PartialDependence(predictor=predictor,
                                  feature_names=feature_names,
                                  categorical_names=categorical_names)

    return explainer.explain(X=X,
                             features=features,
                             kind=kind,
                             percentiles=percentiles,
                             grid_resolution=grid_resolution)


def get_alibi_tree_pd_explanation(predictor: BaseEstimator,
                                  X: np.ndarray,
                                  features:  List[Union[int, Tuple[int, int]]],
                                  percentiles: Tuple[float, float],
                                  grid_resolution: int,
                                  feature_names: Optional[List[str]] = None,
                                  categorical_names: Optional[Dict[int, List[str]]] = None):
    """ Computes `alibi` tree pd explanation. """
    # compute `alibi` explanation
    explainer = TreePartialDependence(predictor=predictor,
                                      feature_names=feature_names,
                                      categorical_names=categorical_names)

    return explainer.explain(X=X,
                             features=features,
                             percentiles=percentiles,
                             grid_resolution=grid_resolution)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [0], [1], [2],
    [(0, 1)], [(0, 2)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'grid_resolution': 10,
        'method': 'brute',
        'kind': 'average'
    }
])
def test_sklearn_numerical(rf_classifier, iris_data, features, params):
    """ Checks `alibi` pd black-box implementation against the `sklearn` implementation for numerical features."""
    rf, _ = rf_classifier
    X_train = iris_data['X_train']

    # compute pd with `alibi`
    exp_alibi = get_alibi_pd_explanation(predictor=rf.predict_proba,
                                         X=X_train,
                                         features=features,
                                         kind=params['kind'],
                                         percentiles=params['percentiles'],
                                         grid_resolution=params['grid_resolution'])

    # compute pd with `sklearn`
    exp_sklearn = partial_dependence(X=X_train, estimator=rf, features=features, **params)

    # compare explanations
    assert np.allclose(exp_alibi.data['pd_values'][0], exp_sklearn['average'])
    assert_feature_values_equal(exp_alibi=exp_alibi, exp_sklearn=exp_sklearn)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [1], [2], [4], [5],
    [(1, 2)], [(2, 4)], [(4, 5)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'method': 'brute',
        'kind': 'average'
    }
])
def test_sklearn_categorical(rf_classifier, adult_data, features, params):
    """ Checks `alibi` pd black-box implementation against the `sklearn` implementation for categorical features."""

    rf, preprocessor = rf_classifier
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('predictor', rf)])

    # subsample data for faster computation
    X_train = adult_data['X_train'][:100]

    # Behaviour depends on sklearn version, See https://github.com/SeldonIO/alibi/pull/940#issuecomment-1623783025
    sklearn_version = version.parse(sklearn.__version__)
    if sklearn_version >= version.parse('1.3.0'):
        categorical_names = adult_data['metadata']['category_map']
        categorical_names = list(categorical_names.keys())
        params.update(categorical_features=categorical_names)
    else:
        params.update(grid_resolution=np.inf)

    # compute `sklearn` explanation
    exp_sklearn = partial_dependence(X=X_train,
                                     estimator=rf_pipeline,
                                     features=features,
                                     **params)

    # update intentionally grid_resolution to check that alibi behaves correctly for categorical features
    params.update(grid_resolution=100)

    # compute alibi explanation
    exp_alibi = get_alibi_pd_explanation(predictor=rf_pipeline.predict_proba,
                                         feature_names=adult_data['metadata']['feature_names'],
                                         categorical_names=adult_data['metadata']['category_map'],
                                         X=X_train,
                                         features=features,
                                         kind=params['kind'],
                                         percentiles=params['percentiles'],
                                         grid_resolution=params['grid_resolution'])

    # compare explanations
    assert np.allclose(exp_alibi.data['pd_values'][0][1], exp_sklearn['average'])
    assert_feature_values_equal(exp_alibi=exp_alibi, exp_sklearn=exp_sklearn)


@pytest.mark.parametrize('predictor', [GradientBoostingClassifier()])
@pytest.mark.parametrize('features', [
    [1], [2], [4], [5],
    [(1, 2)], [(2, 4)], [(4, 5)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'grid_resolution': 30,
        'method': 'recursion',
        'kind': 'average'
    }
])
def test_sklearn_recursion(predictor, binary_data, features, params):
    """ Check `alibi` pd recursion implementation against the `sklearn` implementation. """
    X_train, y_train = binary_data['X_train'], binary_data['y_train']
    predictor = predictor.fit(X_train, y_train)

    # compute `sklearn` explanation
    exp_sklearn = partial_dependence(X=X_train,
                                     estimator=predictor,
                                     features=features,
                                     **params)

    # compute `alibi` explanation
    exp_alibi = get_alibi_tree_pd_explanation(predictor=predictor,
                                              X=X_train,
                                              features=features,
                                              percentiles=params['percentiles'],
                                              grid_resolution=params['grid_resolution'])

    # compare explanations
    assert_feature_values_equal(exp_alibi=exp_alibi, exp_sklearn=exp_sklearn)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [1], [2], [3], [4], [5],
    [(1, 2)], [(2, 3)], [(3, 4)], [(4, 5)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'grid_resolution': 30,
        'method': 'brute',
        'kind': 'both'
    }
])
def test_sklearn_blackbox(rf_classifier, adult_data, features, params):
    """ Checks `alibi` pd black-box implementation against the `sklearn` implementation. """
    predictor, preprocessor = rf_classifier
    predictor_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('predictor', predictor)])

    # subsample dataset for faster computation
    X_train = adult_data['X_train'][:100]

    # compute sklearn explanation
    exp_sklearn = partial_dependence(X=X_train,
                                     estimator=predictor_pipeline,
                                     features=features,
                                     **params)

    # compute alibi explanation
    exp_alibi = get_alibi_pd_explanation(predictor=predictor_pipeline.predict_proba,
                                         feature_names=adult_data['metadata']['feature_names'],
                                         categorical_names=adult_data['metadata']['category_map'],
                                         X=X_train,
                                         features=features,
                                         kind=params['kind'],
                                         percentiles=params['percentiles'],
                                         grid_resolution=params['grid_resolution'])

    # compare explanations
    assert np.allclose(exp_alibi.data['ice_values'][0][1], exp_sklearn['individual'])
    assert np.allclose(exp_alibi.data['pd_values'][0][1], exp_sklearn['average'])
    assert_feature_values_equal(exp_alibi=exp_alibi, exp_sklearn=exp_sklearn)


@pytest.fixture(scope='module')
def explanation(request):
    meta = deepcopy(DEFAULT_META_PD)
    data = deepcopy(DEFAULT_DATA_PD)
    kind = request.param

    meta.update(params={
        'kind': kind,
        'percentiles': (0.0, 1.0),
        'grid_resolution': 2,
        'feature_names': ['f_0', 'f_1', 'f_2', 'f_3'],
        'categorical_names': {2: [0, 1, 2], 3: [0, 1, 2, 3, 4]},
        'target_names': ['c_0']
    })

    data['data'] = {
        'feature_deciles': [
            np.array([-1.189, -1.147, -1.104, -1.06, -1.019, -0.977, -0.934, -0.892, -0.849, -0.807, -0.764]),
            np.array([-0.647, -0.588, -0.528, -0.468, -0.408, -0.348, -0.289, -0.229, -0.169, -0.109, -0.050]),
            None,
            None,
            [
                np.array([-1.189, -1.147, -1.104, -1.062, -1.019, -0.977, -0.934, -0.892, -0.849, -0.807, -0.764]),
                np.array([-0.647, -0.588, -0.528, -0.468, -0.408, -0.348, -0.289, -0.229, -0.169, -0.109, -0.050])
            ],
            [
                np.array([-0.647, -0.588, -0.528, -0.468, -0.408, -0.348, -0.289, -0.229, -0.169, -0.109, -0.050]),
                None
            ],
            [
                None,
                np.array([-0.647, -0.588, -0.528, -0.468, -0.408, -0.348, -0.289, -0.229, -0.169, -0.109, -0.050])
            ],
            [
                None,
                None
            ]
        ],
        'pd_values': [
            np.array([[0.514, 0.469]]),
            np.array([[0.213, 0.771]]),
            np.array([[0.471, 0.515]]),
            np.array([[0.478, 0.506]]),
            np.array([[[0.235, 0.792], [0.193, 0.748]]]),
            np.array([[[0.191, 0.240], [0.742, 0.794]]]),
            np.array([[[0.191, 0.742], [0.240, 0.794]]]),
            np.array([[[0.456, 0.482], [0.503, 0.528]]])
        ],
        'ice_values': [
            np.array([[[0.247, 0.203], [0.781, 0.735]]]),
            np.array([[[0.203, 0.760], [0.223, 0.781]]]),
            np.array([[[0.160, 0.203], [0.781, 0.827]]]),
            np.array([[[0.203, 0.230], [0.753, 0.781]]]),
            np.array([[[[0.247, 0.803], [0.203, 0.760]], [[0.223, 0.781], [0.182, 0.735]]]]),
            np.array([[[[0.160, 0.203], [0.703, 0.760]], [[0.223, 0.277], [0.781, 0.827]]]]),
            np.array([[[[0.160, 0.703], [0.203, 0.760]], [[0.223, 0.781], [0.277, 0.827]]]]),
            np.array([[[[0.160, 0.182], [0.203, 0.230]], [[0.753, 0.781], [0.803, 0.827]]]])
        ],
        'feature_values': [
            np.array([-1.189, -0.764]),
            np.array([-0.647, -0.050]),
            np.array([1., 2.]),
            np.array([0., 4.]),
            [
                np.array([-1.189, -0.764]),
                np.array([-0.647, -0.050])
            ],
            [
                np.array([-0.647, -0.050]),
                np.array([1., 2.])
            ],
            [
                np.array([1., 2.]),
                np.array([-0.647, -0.050])
            ],
            [
                np.array([1., 2.]),
                np.array([0., 4.])
            ]
        ],
        'feature_names': [
            'f_0', 'f_1', 'f_2', 'f_3',
            ('f_0', 'f_1'), ('f_1', 'f_2'), ('f_2', 'f_1'), ('f_2', 'f_3')
        ]
    }
    return Explanation(meta=meta, data=data)


def assert_deciles(xsegments: Optional[List[np.ndarray]] = None,
                   expected_xdeciles: Optional[np.ndarray] = None,
                   ysegments: Optional[List[np.ndarray]] = None,
                   expected_ydeciles: Optional[np.ndarray] = None):
    """ Checks the deciles on the x-axis. """
    if (xsegments is not None) and (expected_xdeciles is not None):
        xdeciles = np.array([segment[0, 0] for segment in xsegments])
        assert np.allclose(xdeciles, expected_xdeciles[1:-1])

    if (ysegments is not None) and (expected_ydeciles is not None):
        ydeciles = np.array([segment[0, 1] for segment in ysegments])
        assert np.allclose(ydeciles, expected_ydeciles[1:-1])


def assert_pd_values(feature_values: np.ndarray, pd_values: np.ndarray, line: plt.Line2D):
    """ Checks if the plotted pd values are correct. """
    x, y = line.get_xydata().T
    assert np.allclose(x, feature_values)
    assert np.allclose(y, pd_values)


def assert_ice_values(feature_values: np.ndarray, ice_values: np.ndarray, lines: List[plt.Line2D]):
    """ Checks if the plotted ice values are correct. """
    for ice_vals, line in zip(ice_values, lines):
        x, y = line.get_xydata().T
        assert np.allclose(x, feature_values)
        assert np.allclose(y, ice_vals)


def assert_pd_ice_values(feature: int, target_idx: int, kind: str, explanation: Explanation, ax: plt.Axes):
    """ Checks if both the plotted pd and ice values are correct. """
    if kind in ['average', 'both']:
        # check the pd values
        line = ax.lines[0] if kind == 'average' else ax.lines[2]
        assert_pd_values(feature_values=explanation.data['feature_values'][feature],
                         pd_values=explanation.data['pd_values'][feature][target_idx],
                         line=line)

    if kind in ['individual', 'both']:
        # check the ice values
        lines = ax.lines if kind == 'individual' else ax.lines[:2]
        assert_ice_values(feature_values=explanation.data['feature_values'][feature],
                          ice_values=explanation.data['ice_values'][feature][target_idx],
                          lines=lines)


@pytest.mark.parametrize('explanation', ['average', 'individual', 'both'], indirect=True)
def test__plot_one_pd_num(explanation):
    feature, target_idx = 0, 0

    _, ax = plt.subplots()
    ax, _ = _plot_one_pd_num(exp=explanation,
                             feature=feature,
                             target_idx=target_idx,
                             center=False,
                             ax=ax)

    # check x-label
    assert ax.get_xlabel() == explanation.data['feature_names'][feature]

    # check deciles on the x-axis
    assert_deciles(xsegments=ax.collections[0].get_segments(),
                   expected_xdeciles=explanation.data['feature_deciles'][feature])

    # check pd and ice values
    assert_pd_ice_values(feature=feature,
                         target_idx=target_idx,
                         kind=explanation.meta['params']['kind'],
                         explanation=explanation,
                         ax=ax)


@pytest.mark.parametrize('explanation', ['average', 'individual', 'both'], indirect=True)
def test__plot_one_pd_cat(explanation):
    feature, target_idx = 2, 0

    _, ax = plt.subplots()
    ax, _ = _plot_one_pd_cat(exp=explanation,
                             feature=feature,
                             target_idx=target_idx,
                             center=False,
                             ax=ax)

    # check x-label
    assert ax.get_xlabel() == explanation.data['feature_names'][feature]

    # check pd and ice values
    assert_pd_ice_values(feature=feature,
                         target_idx=target_idx,
                         kind=explanation.meta['params']['kind'],
                         explanation=explanation,
                         ax=ax)


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test__plot_two_pd_num_num(explanation):
    """ Test the `_plot_two_pd_num_num` function. """
    feature, target_idx = 4, 0

    _, ax = plt.subplots()
    ax, _ = _plot_two_pd_num_num(exp=explanation,
                                 feature=feature,
                                 target_idx=target_idx,
                                 ax=ax)

    assert np.allclose(ax.get_xlim(), explanation.data['feature_values'][feature][0])
    assert np.allclose(ax.get_ylim(), explanation.data['feature_values'][feature][1])

    assert_deciles(xsegments=ax.collections[-2].get_segments(),
                   expected_xdeciles=explanation.data['feature_deciles'][feature][0],
                   ysegments=ax.collections[-1].get_segments(),
                   expected_ydeciles=explanation.data['feature_deciles'][feature][1])

    assert ax.get_xlabel() == explanation.data['feature_names'][feature][0]
    assert ax.get_ylabel() == explanation.data['feature_names'][feature][1]


@pytest.mark.parametrize('feature', [5, 6])
@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test__plot_two_pd_num_cat(feature, explanation):
    """ Test the `__plot_two_pd_num_cat` function. """
    target_idx = 0

    _, ax = plt.subplots()
    ax, _ = _plot_two_pd_num_cat(exp=explanation,
                                 feature=feature,
                                 target_idx=target_idx,
                                 ax=ax)

    feat0, feat1 = explanation.data['feature_names'][feature]
    feature_names = explanation.meta['params']['feature_names']
    categorical_names = explanation.meta['params']['categorical_names']

    num_feat = feat0 if feature_names.index(feat0) not in categorical_names else feat1
    cat_feat = feat0 if feature_names.index(feat0) in categorical_names else feat1

    legend = ax.get_legend()
    legend_title = legend.get_texts()[0].get_text()
    legend_entries = sorted([int(entry.get_text()) for entry in legend.get_texts()[1:]])

    cat_idx = 0 if cat_feat == feat0 else 1
    cat_values = sorted([int(val) for val in explanation.data['feature_values'][feature][cat_idx]])

    assert ax.get_xlabel() == num_feat
    assert legend_title == cat_feat
    assert legend_entries == cat_values

    num_idx = 0 if num_feat == feat0 else 1
    assert_deciles(xsegments=ax.collections[0].get_segments(),
                   expected_xdeciles=explanation.data['feature_deciles'][feature][num_idx])

    pd_values = explanation.data['pd_values'][feature][target_idx]
    if num_idx == 0:
        pd_values = pd_values.T

    for i in range(1, len(explanation.data['feature_values'][feature][cat_idx]) + 1):
        x, y = ax.lines[i].get_xydata().T
        assert np.allclose(x, explanation.data['feature_values'][feature][num_idx])
        assert np.allclose(y, pd_values[i - 1])


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test__plot_two_pd_cat_cat(explanation):
    """ Test the `_plot_two_pd_cat_cat` function. """
    feature, target_idx = 7, 0

    _, ax = plt.subplots()
    ax, _ = _plot_two_pd_cat_cat(exp=explanation,
                                 feature=feature,
                                 target_idx=target_idx,
                                 ax=ax)

    assert np.allclose(ax.images[0].get_array().data, explanation.data['pd_values'][feature])

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    assert xlabel == explanation.data['feature_names'][feature][1]
    assert ylabel == explanation.data['feature_names'][feature][0]

    x_ticklabels = [int(tl.get_text()) for tl in ax.get_xticklabels()]
    y_ticklables = [int(tl.get_text()) for tl in ax.get_yticklabels()]

    feature_names = explanation.meta['params']['feature_names']
    categorical_names = explanation.meta['params']['categorical_names']
    cat_idx0 = feature_names.index(explanation.data['feature_names'][feature][0])
    cat_idx1 = feature_names.index(explanation.data['feature_names'][feature][1])

    expected_x_ticklabels = [categorical_names[cat_idx1][int(val)] for val in
                             explanation.data['feature_values'][feature][1]]
    expected_y_ticklabels = [categorical_names[cat_idx0][int(val)] for val in
                             explanation.data['feature_values'][feature][0]]
    assert np.allclose(expected_x_ticklabels, x_ticklabels)
    assert np.allclose(expected_y_ticklabels, y_ticklables)


def mock_private_plt_function(mocker):
    """ Mocks private specialized plotting functions. """
    mocker.patch('alibi.explainers.partial_dependence._plot_one_pd_num', return_value=(None, None))
    mocker.patch('alibi.explainers.partial_dependence._plot_one_pd_cat', return_value=(None, None))
    mocker.patch('alibi.explainers.partial_dependence._plot_two_pd_num_num', return_value=(None, None))
    mocker.patch('alibi.explainers.partial_dependence._plot_two_pd_num_cat', return_value=(None, None))
    mocker.patch('alibi.explainers.partial_dependence._plot_two_pd_cat_cat', return_value=(None, None))


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_all_features(explanation, mocker):
    """ Test if a PD is plotted for each explained feature. """
    mock_private_plt_function(mocker)
    ax = plot_pd(exp=explanation, features='all')
    assert np.sum(~pd.isna(ax)) == len(explanation.data['feature_names'])


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_oor_feature(explanation):
    """ Test if an error is raised when the feature index is out of range. """
    with pytest.raises(IndexError) as err:
        plot_pd(exp=explanation, features=[len(explanation.data['feature_names'])])
    assert "The `features` indices must be less than the" in str(err.value)


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_unknown_target(explanation):
    """ Test if an error is raised for an unknown target name. """
    with pytest.raises(ValueError) as err:
        plot_pd(exp=explanation, target="unknown")
    assert "Unknown `target` name" in str(err.value)


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_oor_target(explanation):
    """ Test if an error is raised when the target index is out of range. """
    with pytest.raises(IndexError) as err:
        plot_pd(exp=explanation, target=len(explanation.meta['params']['target_names']))
    assert "Target index out of range." in str(err.value)


@pytest.mark.parametrize('n_cols', [2, 3, 4, 5])
@pytest.mark.parametrize('explanation',  ['average'], indirect=True)
def test_plot_pd_n_cols(n_cols, explanation, mocker):
    """ Test if the number figure columns matches the expected one. """
    mock_private_plt_function(mocker)
    ax = plot_pd(exp=explanation, features='all', n_cols=n_cols)
    assert ax.shape[-1] == n_cols


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_ax(explanation):
    """ Test if an error is raised if the number of provided axes is less that the number of features to plot
    the PD for. """
    _, ax = plt.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError) as err:
        plot_pd(exp=explanation, features='all', ax=ax)
    assert "Expected ax to have" in str(err.value)


@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_sharey_all(explanation):
    """ Test if all axes have the same y limits when ``sharey='all'``. """
    axes = plot_pd(exp=explanation, features=[0, 1, 2, 3], n_cols=1, sharey='all')
    assert len(set([ax.get_ylim() for ax in axes.ravel()])) == 1


@pytest.mark.parametrize('n_cols', [1, 2])
@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_sharey_row(n_cols, explanation):
    """ Test if all axes on the same rows have the same y limits and axes on different rows have different y limits
    when ``sharey='row'``. """
    features = [0, 1, 2]
    n_rows = len(features) // n_cols + (len(features) % n_cols != 0)
    axes = plot_pd(exp=explanation, features=features, n_cols=n_cols, sharey='row')
    ylims = []

    for i in range(axes.shape[0]):
        ylim = set([ax.get_ylim() for ax in axes[i] if ax is not None])
        assert len(ylim) == 1
        ylims.append(ylim.pop())

    assert len(set(ylims)) == n_rows


@pytest.mark.parametrize('n_cols', [1, 2, 3])
@pytest.mark.parametrize('explanation', ['average'], indirect=True)
def test_plot_pd_sharey_none(n_cols, explanation):
    """Test if all axes have different y limits when ``sharey=None``. """
    features = [0, 1, 2]
    axes = plot_pd(exp=explanation, features=features, n_cols=n_cols, sharey=None)
    assert len(set([ax.get_ylim() for ax in axes.ravel() if ax is not None])) == len(features)


@pytest.mark.parametrize('n_ice', ['all', 'list', 'int'])
def test_ice_sampling(n_ice):
    """ Test if the ice sampling helper function works properly when the arguments are valid. """
    n_samples, n_values = 100, 10
    ice_vals = np.random.randn(n_values, n_samples)

    if n_ice == 'all':
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice)
        np.testing.assert_allclose(ice_vals, ice_sampled_vals)

    elif n_ice == 'list':
        size = np.random.randint(1, n_samples)
        # needs to be sorted because of the np.unique applied inside the _sample_ice
        n_ice = np.sort(np.random.choice(n_samples, size=size, replace=False)).tolist()
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice)
        np.testing.assert_allclose(ice_vals[:, n_ice], ice_sampled_vals)

    else:
        n_ice = np.random.randint(1, n_samples)
        ice_sampled_vals = _sample_ice(ice_values=ice_vals, n_ice=n_ice)
        assert ice_sampled_vals.shape[1] == n_ice


@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__sample_ice_warning(explanation, caplog):
    """ Test if the sampling function logs a warning when the number of ices is greater that the number
    of instances in the reference dataset and if it returns all ices"""
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    sampled_ice_values = _sample_ice(ice_values=ice_values, n_ice=n_samples + 1)
    assert "`n_ice` is greater than the number of instances in the reference dataset." in caplog.records[0].message
    assert np.allclose(ice_values, sampled_ice_values)


@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__sample_ice_error_negative(explanation):
    """ Test if an error is raise when `n_ice` is negative. """
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    with pytest.raises(ValueError) as err:
        _sample_ice(ice_values=ice_values, n_ice=-2)
    assert "`n_ice` must be an integer grater than 0." == str(err.value)


@pytest.mark.parametrize('n_ice', [1, 2, 4, 8])
@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__sample_ice_error_sample(n_ice, explanation):
    """ Test if the number of sampled ice matches the expectation. """
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    sampled_ice_values = _sample_ice(ice_values=ice_values, n_ice=n_ice)
    assert sampled_ice_values.shape[-1] == n_ice


@pytest.mark.parametrize('n_ice', [[-1, 0, 1], [0, 1, 11]])
@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__sample_ice_error_oor(n_ice, explanation):
    """ Test if an error is raised when the ice indices are out of bounds. """
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    with pytest.raises(ValueError) as err:
        _sample_ice(ice_values=ice_values, n_ice=n_ice)
    assert "Some indices in `n_ice` are out of bounds." in str(err.value)


@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__sample_ice_error_type(explanation):
    """ Test if an error is raise if the `n_ice` is an unknown type. """
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    with pytest.raises(ValueError) as err:
        _sample_ice(ice_values=ice_values, n_ice="unknown")
    assert "Unknown `n_ice` values." in str(err.value)


@pytest.mark.parametrize('explanation', ['both'], indirect=True)
def test__process_pd_ice(explanation):
    """ Test the `center` option for the pd and ice. """
    n_samples, n_values = 10, 5
    ice_values = np.random.randn(n_values, n_samples)
    pd_values = np.mean(ice_values, axis=-1)

    centered_pd_values, centered_ice_values = _process_pd_ice(exp=explanation,
                                                              pd_values=pd_values,
                                                              ice_values=ice_values,
                                                              n_ice='all',
                                                              center=True)
    assert np.isclose(centered_pd_values[0], 0)
    assert np.allclose(centered_ice_values[0], 0)

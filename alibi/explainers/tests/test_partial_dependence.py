import numbers
import re
from typing import List, Tuple

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from alibi.explainers import PartialDependence, TreePartialDependence
from alibi.explainers.partial_dependence import _sample_ice


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
    assert len(exp.feature_names) == len(features)
    assert len(exp.pd_values) == len(features)
    assert len(exp.ice_values) == len(features)
    assert len(exp.feature_values) == len(features)

    for i, f in enumerate(features):
        if isinstance(f, Tuple):
            # check deciles
            assert isinstance(exp.feature_deciles[i], List)
            assert len(exp.feature_deciles[i]) == len(f)
            assert len(exp.feature_deciles[i][0]) == 11
            assert len(exp.feature_deciles[i][1]) == 11

            # check feature_values
            assert isinstance(exp.feature_values[i], List)
            assert len(exp.feature_values[i]) == len(f)
            assert len(exp.feature_values[i][0]) == (len(np.unique(X_train[:, f[0]]))
                                                     if grid_resolution == np.inf else grid_resolution)
            assert len(exp.feature_values[i][1]) == (len(np.unique(X_train[:, f[1]]))
                                                     if grid_resolution == np.inf else grid_resolution)

            # check pd_values
            assert exp.pd_values[i].shape == (num_targets,
                                              len(exp.feature_values[i][0]),
                                              len(exp.feature_values[i][1]))

            # check ice_values
            assert exp.ice_values[i].shape == (num_targets,
                                               num_instances,
                                               len(exp.feature_values[i][0]),
                                               len(exp.feature_values[i][1]))

        else:
            # check feature_deciles
            assert len(exp.feature_deciles[i]) == 11

            # check feature_values
            assert len(exp.feature_values[i]) == (len(np.unique(X_train[:, f]))
                                                  if grid_resolution == np.inf else grid_resolution)

            # check pd_values
            assert exp.pd_values[i].shape == (num_targets, len(exp.feature_values[i]))

            # check ice_value
            assert exp.ice_values[i].shape == (num_targets, num_instances, len(exp.feature_values[i]))


@pytest.mark.parametrize('rf_regressor', [lazy_fixture('boston_data')], indirect=True)
@pytest.mark.parametrize('kind', ['average', 'individual', 'both'])
@pytest.mark.parametrize('features', [
    [0, 1, 2],
    [(0, 1), (0, 2), (1, 2)],
    [0, 1, (0, 1)]
])
def test_blackbox_regression(rf_regressor, boston_data, kind, features):
    """ Test the black-box predictor for a regression function. """
    rf, _ = rf_regressor
    X_train = boston_data['X_train']

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
        np.testing.assert_allclose(exp.feature_values[i], grid_points[i])


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


@pytest.mark.parametrize('n_ice', ['all', 'list', 'int'])
@pytest.mark.parametrize('n_samples, n_values', [(100, 10)])
def test_ice_sampling(n_ice, n_samples, n_values):
    """ Checks if the ice sampling helper function works properly when the arguments are valid. """
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


@pytest.mark.parametrize('n_ice', ['unknown', -10, [-1, 1, 2]])
@pytest.mark.parametrize('n_samples, n_values', [(100, 10)])
def test_ice_sampling_error(n_samples, n_values, n_ice):
    """ Checks if the ice sampling helper function throws an error when the arguments are invalid. """
    ice_vals = np.random.rand(n_values, n_samples)
    with pytest.raises(ValueError):
        _sample_ice(ice_values=ice_vals, n_ice=n_ice)


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
    explainer = PartialDependence(predictor=rf.predict_proba)
    exp_alibi = explainer.explain(X=X_train,
                                  features=features,
                                  kind=params['kind'],
                                  percentiles=params['percentiles'],
                                  grid_resolution=params['grid_resolution'])

    # compute pd with `sklearn`
    exp_sklearn = partial_dependence(X=X_train, estimator=rf, features=features, **params)

    assert np.allclose(exp_alibi.pd_values[0], exp_sklearn['average'])
    if isinstance(features[0], numbers.Integral):
        assert np.allclose(exp_alibi.feature_values[0], exp_sklearn['values'][0])
    else:
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.feature_values[0][i], exp_sklearn['values'][i])


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [1], [2], [4], [5],
    [(1, 2)], [(2, 4)], [(4, 5)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'grid_resolution': np.inf,
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

    # compute `sklearn` explanation
    exp_sklearn = partial_dependence(X=X_train,
                                     estimator=rf_pipeline,
                                     features=features,
                                     **params)

    # update intentionally grid_resolution to check that alibi behaves correctly for categorical features
    params.update(grid_resolution=100)

    # compute alibi explanation
    explainer = PartialDependence(predictor=rf_pipeline.predict_proba,
                                  feature_names=adult_data['metadata']['feature_names'],
                                  categorical_names=adult_data['metadata']['category_map'])
    exp_alibi = explainer.explain(X=X_train,
                                  features=features,
                                  kind=params['kind'],
                                  percentiles=params['percentiles'],
                                  grid_resolution=params['grid_resolution'])

    # compare explanations
    assert np.allclose(exp_alibi.pd_values[0][1], exp_sklearn['average'])
    if isinstance(features[0], numbers.Integral):
        assert np.allclose(exp_alibi.feature_values[0], exp_sklearn['values'][0])
    else:
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.feature_values[0][i], exp_sklearn['values'][i])


@pytest.mark.parametrize('predictor', [GradientBoostingClassifier()])
@pytest.mark.parametrize('features', [
    [1], [2], [4], [5],
    [(1, 2)], [(2, 4)], [(4, 5)]
])
@pytest.mark.parametrize('params', [
    {
        'percentiles': (0, 1),
        'grid_resolution': np.inf,
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
    explainer = TreePartialDependence(predictor=predictor)
    exp_alibi = explainer.explain(X=X_train,
                                  features=features,
                                  percentiles=params['percentiles'],
                                  grid_resolution=params['grid_resolution'])

    # compare explanations
    assert np.allclose(exp_alibi.pd_values[0], exp_sklearn['average'])
    if isinstance(features[0], numbers.Integral):
        assert np.allclose(exp_alibi.feature_values[0], exp_sklearn['values'][0])
    else:
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.feature_values[0][i], exp_sklearn['values'][i])


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
    explainer = PartialDependence(predictor=predictor_pipeline.predict_proba,
                                  feature_names=adult_data['metadata']['feature_names'],
                                  categorical_names=adult_data['metadata']['category_map'])
    exp_alibi = explainer.explain(X=X_train,
                                  features=features,
                                  kind=params['kind'],
                                  percentiles=params['percentiles'],
                                  grid_resolution=params['grid_resolution'])

    # compare explanations
    assert np.allclose(exp_alibi.pd_values[0][1], exp_sklearn['average'])
    assert np.allclose(exp_alibi.ice_values[0][1], exp_sklearn['individual'])

    if isinstance(features[0], numbers.Integral):
        assert np.allclose(exp_alibi.feature_values[0], exp_sklearn['values'][0])
    else:
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.feature_values[0][i], exp_sklearn['values'][i])

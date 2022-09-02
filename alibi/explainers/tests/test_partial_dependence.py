import numbers
import re
from typing import List, Tuple

import numpy as np
import pytest
from alibi.explainers import PartialDependence
from alibi.explainers.partial_dependence import (Kind, Method, ResponseMethod,
                                                 _sample_ice)
from pytest_lazyfixture import lazy_fixture
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


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


@pytest.mark.parametrize('predictor', [RandomForestClassifier()])
def test_unfitted_estimator(predictor):
    """ Checks if raises error for unfitted model. """
    with pytest.raises(NotFittedError) as err:
        PartialDependence(predictor=predictor)
    assert re.search('not fitted yet', err.value.args[0])


@pytest.mark.parametrize('multioutput_classifier', [RandomForestClassifier()], indirect=True)
def test_multioutput_estimator(multioutput_classifier, multioutput_dataset):
    """ Check if raises error for multi-output model"""
    X_train, Y_train = multioutput_dataset['X_train'], multioutput_dataset['Y_train']
    multioutput_classifier.fit(X_train, Y_train)
    with pytest.raises(ValueError) as err:
        PartialDependence(predictor=multioutput_classifier)
    assert re.search('multiclass-multioutput', err.value.args[0].lower())


@pytest.mark.parametrize('response_method', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_response_method(rf_classifier, response_method):
    """ Checks if raises error for unknown `response_method`. """
    predictor, _ = rf_classifier
    with pytest.raises(ValueError) as err:
        PartialDependence(predictor=predictor, response_method=response_method)
    assert re.search("``response_method=\'\w+\'`` is invalid", err.value.args[0].lower())  # noqa: W605


@pytest.mark.parametrize('rf_regressor', [lazy_fixture('boston_data')], indirect=True)
@pytest.mark.parametrize('response_method', [
    ResponseMethod.DECISION_FUNCTION.value,
    ResponseMethod.PREDICT_PROBA.value
])
def test_estimator_response_method(rf_regressor, response_method):
    """ Checks if raises error for a regressor with a ``response_method!='auto'``. """
    predictor, _ = rf_regressor
    with pytest.raises(ValueError) as err:
        PartialDependence(predictor=predictor, response_method=response_method)
    assert re.search('The `response_method` parameter must be ``None`` for regressor.', err.value.args[0])


@pytest.mark.parametrize('method', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_method(rf_classifier, method):
    """ Checks if raises error for unknown `method`. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor, response_method='predict_proba')
    with pytest.raises(ValueError) as err:
        explainer._sklearn_params_sanity_checks(method=method)
    assert re.search("``method=\'\w+\'`` is invalid", err.value.args[0].lower())  # noqa: W605


@pytest.mark.parametrize('kind', ['unknown'])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unknown_kind(rf_classifier, kind):
    """ Checks if raises error for unknown `kind`. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor, response_method='predict_proba')
    with pytest.raises(ValueError) as err:
        explainer._sklearn_params_sanity_checks(kind=kind)
    assert re.search("``kind=\'\w+\'`` is invalid", err.value.args[0].lower())  # noqa: W605


@pytest.mark.parametrize('kind', [Kind.INDIVIDUAL, Kind.BOTH])
@pytest.mark.parametrize('method', [Method.RECURSION])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_kind_method(rf_classifier, kind, method):
    """ Checks if raises error when ``method='recursion'`` and ``kind!='average'``. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor, response_method='decision_function')
    with pytest.raises(ValueError) as err:
        explainer._sklearn_params_sanity_checks(kind=kind, method=method)
    assert re.search("then the `kind` value must be ", err.value.args[0].lower())


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
def test_unsupported_method_recursion(rf_classifier):
    """ Checks if raises error when the ``method='recursion'`` for a classifier which does not support it. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor, response_method='decision_function')
    with pytest.raises(ValueError) as err:
        explainer._sklearn_params_sanity_checks(method=Method.RECURSION)
    assert re.search("``method='recursion'`` is only supported by", err.value.args[0].lower())


@pytest.mark.parametrize('predictor', [GradientBoostingClassifier()])
def test_method_recursion_response_method_predict_proba(predictor, iris_data):
    """ Checks if raises error when ``method='recursion'`` for a classifier which supports it and when
    the ``response_method='predict_proba'``. """
    X_train, y_train = iris_data['X_train'], iris_data['y_train']
    predictor.fit(X_train, y_train)

    explainer = PartialDependence(predictor=predictor, response_method='predict_proba')
    with pytest.raises(ValueError) as err:
        explainer._sklearn_params_sanity_checks(method='recursion')
    assert re.search('then the `response_method` value must be', err.value.args[0].lower())


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('features', [
    [(0, 1, 2)],
    [0, (0, 1), (0, 1, 2)],
    [0, 1, 2, (0, 1, 2)],
    [0, 1, tuple()]
])
def test_num_features(rf_classifier, iris_data, features):
    """ Checks if raises error when a requested partial dependence for a tuple containing more than two features or
    fewer than one. """
    predictor, _ = rf_classifier
    explainer = PartialDependence(predictor=predictor,
                                  response_method='predict_proba',
                                  feature_names=list(range(iris_data['X_train'].shape[1])))
    with pytest.raises(ValueError):
        explainer._features_sanity_checks(features=features)


@pytest.mark.parametrize('rf_classifier', [lazy_fixture('iris_data')], indirect=True)
@pytest.mark.parametrize('grid_resolution', [3, 5, np.inf])
@pytest.mark.parametrize('features', [
    [0, 1, (0, 1), (1, 2)]
])
def test_explanation_numerical_shapes(rf_classifier, iris_data, grid_resolution, features):
    """ Checks the correct shapes of the arrays contained in the explanation object for numerical features. """
    predictor, _ = rf_classifier
    X_train, y_train = iris_data['X_train'][:30], iris_data['y_train'][:30]
    num_targets = len(np.unique(y_train))
    num_instances = len(X_train)

    explainer = PartialDependence(predictor=predictor, response_method='predict_proba')
    exp = explainer.explain(X=X_train, features=features, grid_resolution=grid_resolution, kind='both')

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
                      grid_resolution=10,
                      method='brute',
                      kind=kind)


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
                      grid_resolution=10,
                      method='brute',
                      kind=kind)


@pytest.mark.parametrize('use_int', [False, True])
@pytest.mark.parametrize('rf_classifier', [lazy_fixture('adult_data')], indirect=True)
def test_grid_points(adult_data, rf_classifier, use_int):
    """ Checks whether the grid points provided are used for computing the partial dependencies. """
    rf, preprocessor = rf_classifier
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('predictor', rf)])

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
    explainer = PartialDependence(predictor=rf_pipeline,
                                  response_method='predict_proba',
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


@pytest.mark.parametrize('response_method', ['predict_proba'])
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('binary_data')], indirect=True)
def test_binary_classifier_two_targets(lr_classifier, binary_data, response_method):
    """ Checks that for a classifier which has `predict_proba`  for which the ``response_method='predict_proba'`,
    the partial dependence has two targets. """
    lr, _ = lr_classifier
    X_train = binary_data['X_train']

    explainer = PartialDependence(predictor=lr, response_method=response_method)
    exp = explainer.explain(X=X_train, method='brute', kind='both')

    for pd, ice in zip(exp.pd_values, exp.ice_values):
        assert pd.shape[0] == 2
        assert ice.shape[0] == 2


@pytest.mark.parametrize('svc_classifier', [lazy_fixture('binary_data')], indirect=True)
def test_binary_classifier_one_target(svc_classifier, binary_data):
    """ Checks that for a classifier which does not have `predict_proba` and for which the ``response_method='auto'``,
    the partial dependence has only one target. """
    svc, _ = svc_classifier
    X_train = binary_data['X_train']

    explainer = PartialDependence(predictor=svc)
    exp = explainer.explain(X=X_train, kind='both')

    for pd, ice in zip(exp.pd_values, exp.ice_values):
        assert pd.shape[0] == 1
        assert ice.shape[0] == 1


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
    """ Checks `alibi` pd implementation against the `sklearn` implementation for numerical features."""
    rf, _ = rf_classifier
    X_train = iris_data['X_train']

    # compute pd with `alibi`
    explainer = PartialDependence(predictor=rf, response_method='predict_proba')
    exp_alibi = explainer.explain(X=X_train, features=features, **params)

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
    """ Checks `alibi` pd implementation against the `sklearn` implementation for categorical features."""
    rf, preprocessor = rf_classifier
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('predictor', rf)])
    X_train = adult_data['X_train'][:100]

    # compute `sklearn` explanation
    exp_sklearn = partial_dependence(X=X_train, estimator=rf_pipeline, features=features, **params)

    # update intentionally grid_resolution to check that alibi behaves correctly for categorical features
    params.update(grid_resolution=100)

    # compute alibi explanation
    explainer = PartialDependence(predictor=rf_pipeline,
                                  response_method='predict_proba',
                                  feature_names=adult_data['metadata']['feature_names'],
                                  categorical_names=adult_data['metadata']['category_map'])
    exp_alibi = explainer.explain(X=X_train, features=features, **params)

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
    exp_sklearn = partial_dependence(X=X_train, estimator=predictor, features=features, **params)

    # compute `alibi` explanation
    explainer = PartialDependence(predictor=predictor, response_method='decision_function')
    exp_alibi = explainer.explain(X=X_train, features=features, **params)

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
    rf, preprocessor = rf_classifier
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('predictor', rf)])
    X_train = adult_data['X_train'][:100]

    # compute sklearn explanation
    exp_sklearn = partial_dependence(X=X_train, estimator=rf_pipeline, features=features, **params)

    # compute alibi explanation
    explainer = PartialDependence(predictor=rf_pipeline.predict_proba,
                                  response_method='predict_proba',
                                  feature_names=adult_data['metadata']['feature_names'],
                                  categorical_names=adult_data['metadata']['category_map'])
    exp_alibi = explainer.explain(X=X_train, features=features, **params)

    # compare explanations
    assert np.allclose(exp_alibi.pd_values[0][1], exp_sklearn['average'])
    assert np.allclose(exp_alibi.ice_values[0][1], exp_sklearn['individual'])

    if isinstance(features[0], numbers.Integral):
        assert np.allclose(exp_alibi.feature_values[0], exp_sklearn['values'][0])
    else:
        for i in range(len(exp_sklearn['values'])):
            assert np.allclose(exp_alibi.feature_values[0][i], exp_sklearn['values'][i])

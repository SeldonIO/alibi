import pytest
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, max_error
from alibi.explainers import PermutationImportance


@pytest.fixture(scope='module')
def dataset():
    return {
        'classification': {
            'y_true': np.random.randint(0, 2, size=(100, )),
            'y_pred': np.random.randint(0, 2, size=(100, )),
        },
        'regression': {
            'y_true': np.random.randn(100),
            'y_pred': np.random.randn(100),
        },
        'sample_weight': np.random.rand(100)
    }


def test_provided_metrics():
    """ Test if the initialization raises an error when neither the loss and the score functions are provided. """
    with pytest.raises(ValueError) as err:
        PermutationImportance(predictor=lambda x: x)

    assert "At least one loss function or a score function must be provided." in str(err.value)


def test_init_metrics_list():
    """ Test if the metrics initialization raises an error when the metrics are provided through a list but the
    elements within the list are not strings. """
    with pytest.raises(ValueError) as err:
        PermutationImportance(predictor=lambda x: x, loss_fns=[lambda x: x])

    assert re.search("The .+ inside .+_fns must be of type `str`.", str(err.value))


def test_init_metrics_unknown():
    """ Test if the metrics initialization raises an error when the metrics provided are unknown. """
    with pytest.raises(ValueError) as err:
        PermutationImportance(predictor=lambda x: x, loss_fns=['unknown'])

    assert re.search("Unknown .+ name.", str(err.value))


def test_compute_metric_unsupported_sample_weight(dataset, caplog):
    """ Test if a warning message is displayed when the metric does not support sample weight. """
    y_true = dataset['regression']['y_true']
    y_pred = dataset['regression']['y_pred']
    sample_weight = dataset['sample_weight']

    metric = PermutationImportance._compute_metric(metric_fn=max_error,
                                                   y=y_true,
                                                   y_hat=y_pred,
                                                   sample_weight=sample_weight)

    expected_max_error = max_error(y_true=y_true,
                                   y_pred=y_pred)

    assert np.isclose(expected_max_error, metric)
    assert re.search("The loss function .+ does not support argument `sample_weight`.", caplog.text)


@pytest.mark.parametrize('use_sample_weight', [True, False])
def test_compute_metric(use_sample_weight, dataset):
    """ Test if the computation of the metric is correct. """
    y_true = dataset['classification']['y_true']
    y_pred = dataset['classification']['y_pred']
    sample_weight = dataset['sample_weight']

    weighted_metric = PermutationImportance._compute_metric(metric_fn=accuracy_score,
                                                            y=y_true,
                                                            y_hat=y_pred,
                                                            sample_weight=sample_weight if use_sample_weight else None)

    expected_weighted_metric = accuracy_score(y_true=y_true,
                                              y_pred=y_pred,
                                              sample_weight=sample_weight if use_sample_weight else None)

    assert np.allclose(expected_weighted_metric, weighted_metric)


def test_compute_metric_error():
    """ Test if an error is raised when the metric function does not have the `y_true` or
    `y_pred` or `y_score` arguments. """
    with pytest.raises(ValueError) as err1:
        PermutationImportance._compute_metric(metric_fn=lambda y, y_pred: None, y=None, y_hat=None)

    with pytest.raises(ValueError) as err2:
        PermutationImportance._compute_metric(metric_fn=lambda y_true, y_hat: None, y=None, y_hat=None)

    assert "The `scoring` function must have the argument `y_true` in its definition." in str(err1.value)
    assert "The `scoring` function must have the argument `y_pred` or `y_score` in its definition." in str(err2.value)


def test_compute_metrics(dataset):
    """ Test if the computation of multiple metrics is correct. """
    y_true = dataset['classification']['y_true']
    y_pred = dataset['classification']['y_pred']
    sample_weight = dataset['sample_weight']

    metrics_fns = {
        'accuracy': accuracy_score,
        'f1': f1_score,
    }

    metrics = PermutationImportance._compute_metrics(metric_fns=metrics_fns,
                                                     y=y_true,
                                                     y_hat=y_pred,
                                                     sample_weight=sample_weight)

    expected_accuracy = accuracy_score(y_true=y_true,
                                       y_pred=y_pred,
                                       sample_weight=sample_weight)

    expected_f1 = f1_score(y_true=y_true,
                           y_pred=y_pred,
                           sample_weight=sample_weight)

    assert np.isclose(expected_accuracy, metrics['accuracy'])
    assert np.isclose(expected_f1, metrics['f1'])


@pytest.mark.parametrize('lower_is_better', [True, False])
def test_compute_importances_exact_ratio(lower_is_better):
    """ Test if the computation of importance scores for multiple metrics is correct when `method='exact'`
    and `kind='ratio'`. """
    metric_orig = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)],
    }

    metric_permuted = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)],
    }

    feature_importances = PermutationImportance._compute_importances(metric_orig=metric_orig,
                                                                     metric_permuted=metric_permuted,
                                                                     kind='ratio',
                                                                     lower_is_better=lower_is_better)

    for mn in metric_orig:
        expected_importance = metric_permuted[mn][0] / metric_orig[mn][0]

        if not lower_is_better:
            expected_importance = 1. / expected_importance

        assert np.isclose(expected_importance, feature_importances[mn])


@pytest.mark.parametrize('lower_is_better', [True, False])
def test_compute_importances_exact_difference(lower_is_better):
    """ Test if the computation of importance scores for multiple metrics is correct when `method='exact`
    and `kind='difference'`. """
    metric_orig = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)]
    }

    metric_permuted = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)]
    }

    feature_importances = PermutationImportance._compute_importances(metric_orig=metric_orig,
                                                                     metric_permuted=metric_permuted,
                                                                     kind='difference',
                                                                     lower_is_better=lower_is_better)

    sign = 1 if lower_is_better else -1
    for mn in metric_orig:
        expected_importance = sign * (metric_permuted[mn][0] - metric_orig[mn][0])
        assert np.isclose(expected_importance, feature_importances[mn])


@pytest.mark.parametrize('lower_is_better', [True, False])
def test_compute_importances_estimate_ratio(lower_is_better):
    """ Test if the computation of importance scores for multiple metrics is correct when `method='exact'`
    and `kind='ratio'.` """
    metric_orig = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)],
    }

    metric_permuted = {
        'metric_1': np.random.uniform(1, 2, size=(4, )).tolist(),
        'metric_2': np.random.uniform(1, 2, size=(4, )).tolist()
    }

    feature_importances = PermutationImportance._compute_importances(metric_orig=metric_orig,
                                                                     metric_permuted=metric_permuted,
                                                                     kind='ratio',
                                                                     lower_is_better=lower_is_better)

    for mn in metric_orig:
        samples = np.array([mp / metric_orig[mn][0] for mp in metric_permuted[mn]])

        if not lower_is_better:
            samples = 1 / samples

        mean, std = samples.mean(), samples.std()
        assert np.allclose(mean, feature_importances[mn]['mean'])
        assert np.allclose(std, feature_importances[mn]['std'])
        assert np.allclose(samples, feature_importances[mn]['samples'])


@pytest.mark.parametrize('lower_is_better', [True, False])
def test_compute_importances_estimate_difference(lower_is_better):
    """ Test if the computation of importance scores for multiple metrics is correct when `method='exact'`
    and `kind='difference'`. """
    metric_orig = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)],
    }

    metric_permuted = {
        'metric_1': np.random.uniform(1, 2, size=(4, )).tolist(),
        'metric_2': np.random.uniform(1, 2, size=(4, )).tolist(),
    }

    feature_importance = PermutationImportance._compute_importances(metric_orig=metric_orig,
                                                                    metric_permuted=metric_permuted,
                                                                    kind='difference',
                                                                    lower_is_better=lower_is_better)

    for mn in metric_orig:
        samples = np.array([mp - metric_orig[mn][0] for mp in metric_permuted[mn]])

        if not lower_is_better:
            samples = -samples

        mean, std = samples.mean(), samples.std()
        assert np.allclose(mean, feature_importance[mn]['mean'])
        assert np.allclose(std, feature_importance[mn]['std'])
        assert np.allclose(samples, feature_importance[mn]['samples'])


def test_compute_exact(mocker):
    """ Test if the exact computation generates the expected `N x (N - 1)` instances. """
    X = np.array([
        [0, 3],
        [1, 4],
        [2, 5],
    ])
    y = np.array([0, 1, 2])
    y_hat = np.array([0, 1, 2])

    X_expected = np.array([
        [1, 3],
        [2, 3],
        [0, 4],
        [2, 4],
        [0, 5],
        [1, 5],
    ])
    y_expected = np.array([0, 0, 1, 1, 2, 2])
    y_hat_expected = np.array([1, 2, 0, 2, 0, 1])  # first column in X_expected

    score_fns = {'accuracy': accuracy_score}
    score_orig = {"accuracy": [accuracy_score(y_true=y, y_pred=y_hat)]}
    pfi = PermutationImportance(predictor=lambda x: x[:, 0],
                                score_fns=score_fns)

    mock_pred = mocker.patch.object(pfi, 'predictor', wraps=pfi.predictor)
    mock_metrics = mocker.patch.object(PermutationImportance, '_compute_metrics', wraps=pfi._compute_metrics)
    feature_importances = pfi._compute_exact(X=X,
                                             y=y,
                                             kind='difference',
                                             sample_weight=None,
                                             features=0,
                                             loss_orig={},
                                             score_orig=score_orig)

    X_full = np.concatenate([args_pred[0] for args_pred, _ in mock_pred.call_args_list])
    assert np.allclose(X_full, X_expected)

    _, kwargs_metrics = mock_metrics.call_args
    assert np.allclose(y_expected, kwargs_metrics['y'])
    assert np.allclose(y_hat_expected, kwargs_metrics['y_hat'])
    assert np.isclose(feature_importances['accuracy'], 1)


@pytest.mark.parametrize('n_repeats', [3, 4, 5])
def test_compute_estimate(n_repeats, mocker):
    """ Test if the estimate computation generates the expected samples and if the construction of the
     sample is correct. """
    X = np.array([
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 10]
    ])
    y = np.array([1, 2, 3, 4, 5])
    pfi = PermutationImportance(predictor=lambda x: x[:, 0],
                                score_fns={'accuracy': accuracy_score})

    mock = mocker.patch.object(PermutationImportance, '_compute_metrics', wraps=pfi._compute_metrics)
    feature_importances = pfi._compute_estimate(X=X,
                                                y=y,
                                                kind='difference',
                                                n_repeats=n_repeats,
                                                sample_weight=None,
                                                features=0,
                                                loss_orig={},
                                                score_orig={'accuracy': [1.]})

    for _, kwargs in mock.call_args_list:
        y_call, y_hat_call = kwargs['y'], kwargs['y_hat']
        assert len(y_call) == len(y_hat_call)
        assert len(y_call) % 2 == 0

        start, middle, end = 0, len(y_call) // 2, len(y_call)
        fh, sh = np.s_[start:middle], np.s_[middle:end]
        assert np.allclose(y_call[fh], y_hat_call[sh])
        assert np.allclose(y_call[sh], y_hat_call[fh])
        assert np.all(np.isin(y_call, y))

    assert mock.call_count == 2 * n_repeats
    assert np.isclose(feature_importances['accuracy']['mean'], 1)
    assert np.isclose(feature_importances['accuracy']['std'], 0)
    assert len(feature_importances['accuracy']['samples']) == n_repeats
    assert np.allclose(feature_importances['accuracy']['samples'], 1)


@pytest.mark.parametrize('target_col', [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize('method', ['exact', 'estimate'])
@pytest.mark.parametrize('kind', ['difference'])
def test_explain_exact(target_col, method, kind):
    """ Integration test to check correctness. """
    X = np.arange(90).reshape(9, 10).T
    y = np.arange(10) + 10 * target_col

    pfi = PermutationImportance(predictor=lambda x: x[:, target_col], score_fns=["accuracy"])
    exp = pfi.explain(X=X, y=y, method=method, kind=kind)

    if method == 'exact':
        feature_importance = exp.data['feature_importance'][0]
        best_idx = np.argmax(feature_importance)

        assert best_idx == target_col
        assert np.isclose(feature_importance[best_idx], 1.)
        assert np.allclose(np.delete(feature_importance, best_idx), 0)

    else:
        feature_importance = exp.data['feature_importance'][0]
        mean = [fi['mean'] for fi in feature_importance]
        std = [fi['std'] for fi in feature_importance]
        samples = [fi['samples'] for fi in feature_importance]

        best_idx = np.argmax(mean)
        assert best_idx == target_col
        assert np.isclose(mean[best_idx], 1)
        assert np.allclose(np.delete(mean, best_idx), 0)
        assert np.allclose(std, 0)
        assert np.allclose(samples[best_idx], 1)
        assert np.all(np.allclose(s, 0) for s in np.delete(samples, best_idx))

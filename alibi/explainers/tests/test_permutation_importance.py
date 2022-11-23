import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from sklearn.metrics import accuracy_score, f1_score, max_error

from alibi.api.defaults import (DEFAULT_DATA_PERMUTATION_IMPORTANCE,
                                DEFAULT_META_PERMUTATION_IMPORTANCE)
from alibi.api.interfaces import Explanation
from alibi.explainers import PermutationImportance, plot_permutation_importance


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


@pytest.fixture(scope='module')
def exp_estimate():
    """ Creates an estimate explanation object. """
    meta = deepcopy(DEFAULT_META_PERMUTATION_IMPORTANCE)
    data = deepcopy(DEFAULT_DATA_PERMUTATION_IMPORTANCE)

    meta['params'] = {
        'feature_names': ['f_0', 'f_2', 'f_4'],
        'method': 'estimate',
        'kind': 'ratio',
        'n_repeats': 3,
        'sample_weight': None
    }

    data['feature_names'] = ['f_0', 'f_2', 'f_4']
    data['metric_names'] = ['mean_squared_error', 'mean_absolute_error', 'r2']
    data['feature_importance'] = [
        [
            {
                'mean': 2627.702,
                'std': 189.985,
                'samples': np.array([2399.303, 2864.445, 2619.357])
            },
            {
                'mean': 4661.244,
                'std': 161.543,
                'samples': np.array([4530.869, 4563.963, 4888.901])
            },
            {
                'mean': 2012.644,
                'std': 214.766,
                'samples': np.array([2312.335, 1905.535, 1820.063])
            }
        ],
        [
            {
                'mean': 37.519,
                'std': 2.382,
                'samples': np.array([39.548, 34.174, 38.834])
            },
            {
                'mean': 61.574,
                'std': 5.984,
                'samples': np.array([61.956, 68.705, 54.061])
            },
            {
                'mean': 47.815,
                'std': 2.525,
                'samples': np.array([50.765, 48.082, 44.597])
            }
        ],
        [
            {
                'mean': 0.219,
                'std': 0.015,
                'samples': np.array([0.200, 0.238, 0.218])
            },
            {
                'mean': 0.403,
                'std': 0.013,
                'samples': np.array([0.392, 0.395, 0.423])
            },
            {
                'mean': 0.177,
                'std': 0.018,
                'samples': np.array([0.203, 0.167, 0.160])
            }
        ]
    ]
    return Explanation(meta=meta, data=data)


@pytest.fixture(scope='module')
def exp_exact():
    """ Creates an exact explanation object. """
    meta = deepcopy(DEFAULT_META_PERMUTATION_IMPORTANCE)
    data = deepcopy(DEFAULT_DATA_PERMUTATION_IMPORTANCE)

    meta['params'] = {
        'feature_names': ['f_0', 'f_1', 'f_2', 'f_3', 'f_4'],
        'method': 'exact',
        'kind': 'difference',
        'n_repeats': 50,
        'sample_weight': None
    }

    data['feature_names'] = ['f_0', 'f_2', 'f_4']
    data['metric_names'] = ['mean_squared_error', 'mean_absolute_error', 'r2']
    data['feature_importance'] = [
        [3088.364, 7939.829, 4702.336],
        [35.848, 63.491, 50.199],
        [0.212, 0.539, 0.415]
    ]
    return Explanation(meta=meta, data=data)


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_features_all(exp):
    """ Test if all the features are plotted if `features='all'`. """
    axes = plot_permutation_importance(exp=exp, features='all').ravel()
    feature_names = set(exp.data['feature_names'])

    for ax in axes:
        yticklabels = set([txt.get_text() for txt in ax.get_yticklabels()])
        assert feature_names == yticklabels


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_metric_names_all(exp):
    """ Test if all the metrics are plotted if `metric_names='all'`. """
    axes = plot_permutation_importance(exp=exp, metric_names='all').ravel()
    assert len(axes) == len(exp.data['metric_names'])

    titles = [ax.get_title() for ax in axes]
    assert titles == exp.data['metric_names']


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_feature_index_oor(exp):
    """ Test if an error is raised when the feature index is out of range. """
    feature_idx = len(exp.data['feature_names'])
    with pytest.raises(IndexError) as err:
        plot_permutation_importance(exp=exp, features=[feature_idx])
    assert "The `features` indices must be less than the ``len(feature_names)" in str(err.value)


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_metric_name_unknown(exp):
    """ Test if an error is raised when the metric name is unknown. """
    metric_name = 'unknown'
    with pytest.raises(ValueError) as err:
        plot_permutation_importance(exp=exp, metric_names=[metric_name])
    assert "Unknown metric name." in str(err.value)


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_metric_index_oor(exp):
    """ Test if an error is raised when the metric index is out of range. """
    metric_idx = len(exp.data['metric_names'])
    with pytest.raises(IndexError) as err:
        plot_permutation_importance(exp=exp, metric_names=[metric_idx])
    assert "Metric name index out of range." in str(err.value)


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
@pytest.mark.parametrize('n_cols', [1, 2, 3])
def test_plot_pi_n_cols(exp, n_cols):
    """ Test if the number of figure columns matches the expected one. """
    ax = plot_permutation_importance(exp=exp, n_cols=n_cols)
    assert ax.shape[-1] == n_cols


@pytest.mark.parametrize('exp', [lazy_fixture('exp_exact'), lazy_fixture('exp_estimate')])
def test_plot_pi_ax(exp):
    """ Test if an error is raised when the number of axes provided is less than the number
    of targets to be plotted. """
    num_metrics = len(exp.data['metric_names']) - 1
    _, ax = plt.subplots(nrows=1, ncols=num_metrics)

    with pytest.raises(ValueError) as err:
        plot_permutation_importance(exp=exp, ax=ax)
    assert "Expected ax to have" in str(err.value)


@pytest.mark.parametrize('sort, top_k', [(False, None), (True, 1), (True, 2), (True, 3)])
@pytest.mark.parametrize('metric_names', [[0], [1], [0, 2, 1]])
@pytest.mark.parametrize('features', [[0], [1], [0, 2], [2, 0, 1]])
@pytest.mark.parametrize('exp', [lazy_fixture('exp_estimate')])
def test_plot_pi_hbar_values(sort, top_k, metric_names, features, exp):
    """ Test if the horizontal bar plot displays the correct values for an exact explanation. """
    axes = plot_permutation_importance(exp=exp,
                                       features=features,
                                       metric_names=metric_names,
                                       sort=sort,
                                       top_k=top_k).ravel()

    method = exp.meta['params']['method']

    for ax, metric in zip(axes, metric_names):
        if method == 'exact':
            datavalues = ax.containers[0].datavalues
            expected_datavalues = np.array([exp.data['feature_importance'][metric][ft] for ft in features])
        else:
            datavalues = ax.containers[1].datavalues
            segments = ax.containers[0].get_children()[0].get_segments()
            stdevs = np.array([(seg[1, 0] - seg[0, 0]) / 2 for seg in segments])
            expected_datavalues = np.array([exp.data['feature_importance'][metric][ft]['mean'] for ft in features])
            expected_stdevs = np.array([exp.data['feature_importance'][metric][ft]['std'] for ft in features])

        feature_names = [txt.get_text() for txt in ax.get_yticklabels()]
        expected_feature_names = [exp.data['feature_names'][ft] for ft in features]

        if sort:
            sorted_idx = np.argsort(expected_datavalues)[::-1][:top_k]
            expected_datavalues = expected_datavalues[sorted_idx]
            expected_feature_names = [expected_feature_names[i] for i in sorted_idx]

            if method == 'estimate':
                expected_stdevs = expected_stdevs[sorted_idx]

        assert np.allclose(datavalues, expected_datavalues)
        assert feature_names == expected_feature_names
        assert ax.get_title() == exp.data['metric_names'][metric]

        if method == 'estimate':
            assert np.allclose(stdevs, expected_stdevs)

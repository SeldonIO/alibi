import pytest
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, max_error
from alibi.explainers import PermutationImportance


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


def test_compute_metric_unsupported_sample_weight(caplog):
    """ Test if a warning message is displayed when the metric does not support sample weight. """
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    sample_weight = np.random.rand(100)

    metric = PermutationImportance._compute_metric(metric_fn=max_error,
                                                   y=y_true,
                                                   y_hat=y_pred,
                                                   sample_weight=sample_weight)

    expected_max_error = max_error(y_true=y_true,
                                   y_pred=y_pred)

    assert np.isclose(expected_max_error, metric)
    assert re.search("The loss function .+ does not support argument `sample_weight`.", caplog.text)


@pytest.mark.parametrize('use_sample_weight', [True, False])
def test_compute_metric(use_sample_weight):
    """ Test if the computation of the metric is correct. """
    y_true = np.random.randint(0, 2, size=(100, ))
    y_pred = np.random.randint(0, 2, size=(100, ))
    sample_weight = np.random.rand(100)

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


def test_compute_metrics():
    """ Test if the computation of multiple metrics is correct. """
    y_true = np.random.randint(0, 2, size=(100, ))
    y_pred = np.random.randint(0, 2, size=(100, ))
    sample_weight = np.random.rand(100)
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
    """ Test if the computation of importance scores for multiple metrics is correct for `method='exact'`
    and `kind='ratio'`. """
    metric_orig = {
        'metric_1': np.random.uniform(1, 2),
        'metric_2': np.random.uniform(1, 2),
    }

    metric_permuted = {
        'metric_1': [np.random.uniform(1, 2)],
        'metric_2': [np.random.uniform(1, 2)],
    }

    feature_importances = PermutationImportance._compute_importances(metric_orig=metric_orig,
                                                                     metric_permuted=metric_permuted,
                                                                     kind='ratio',
                                                                     lower_is_better=lower_is_better)
    expected_feature_importances = {mn: metric_permuted[mn][0] / metric_orig[mn] for mn in metric_orig}

    if not lower_is_better:
        expected_feature_importances = {k: 1/v for k, v in expected_feature_importances.items()}

    for mn in feature_importances:
        assert np.allclose(expected_feature_importances[mn], feature_importances[mn])


@pytest.mark.parametrize('lower_is_better', [True, False])
def test_compute_importances_exact_difference(lower_is_better):
    """ Test if the computation of importance scores for multiple metrics is correct for `method='exact`
    and `kind='difference'`. """
    metric_orig = {
        'metric_1': np.random.uniform(1, 2),
        'metric_2': np.random.uniform(1, 2)
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
    expected_feature_importances = {mn: sign * (metric_permuted[mn][0] - metric_orig[mn]) for mn in metric_orig}

    for mn in feature_importances:
        assert np.allclose(expected_feature_importances[mn], feature_importances[mn])

from collections import namedtuple
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
from alibi.explainers.experimental.counterfactuals import (
    _WachterCounterfactual,
    _validate_wachter_loss_spec,
    _select_features
)
from alibi.explainers.exceptions import CounterfactualError

bounds = namedtuple('bounds', 'lb ub')


@pytest.mark.parametrize('lams', [np.logspace(-1, -3, num=3)])  # [1e-1, 1e-2, 1e-3]
@pytest.mark.parametrize('cf_found,expected_bounds', [(np.array([1, 1, 1]), bounds(lb=1e-1, ub=1.)),  # 10x lower bound
                                                      (np.array([0, 1, 1]), bounds(lb=1e-3, ub=1e-1)),  # standard
                                                      (np.array([0, 1, 0]), bounds(lb=1e-2, ub=1e-1)),  # impossible?
                                                      (np.array([1, 0, 0]), bounds(lb=1e-1, ub=1.))])  # 10x lower bound
def test_compute_lam_bounds(cf_found, lams, expected_bounds):
    bounds = _WachterCounterfactual.compute_lam_bounds(cf_found, lams)
    lb, ub = bounds.lb, bounds.ub
    assert_almost_equal(lb, expected_bounds.lb)
    assert_almost_equal(ub, expected_bounds.ub)


@pytest.mark.parametrize('predictor_type', ['whitebox', 'blackbox'])
@pytest.mark.parametrize('loss_spec', [{'prediction': {'kwargs': {}}}])
def test_no_distance__validate_wachter_loss_spec(loss_spec, predictor_type):
    with pytest.raises(CounterfactualError) as excinfo:
        _validate_wachter_loss_spec(loss_spec, predictor_type)
    assert 'Expected loss_spec to have key' in str(excinfo.value)


@pytest.mark.parametrize('predictor_type', ['blackbox'])
@pytest.mark.parametrize('loss_spec', [{'distance': {'kwargs': {}}}])
def test_no_grad_fn__validate_wachter_loss_spec(loss_spec, predictor_type):
    with pytest.raises(CounterfactualError) as excinfo:
        _validate_wachter_loss_spec(loss_spec, predictor_type)
    assert 'When one of the loss terms' in str(excinfo.value)


@pytest.mark.parametrize('predictor_type', ['blackbox'])
@pytest.mark.parametrize('loss_spec', [{'distance': {'pred_out_grad_fcn': None,
                                                     'pred_out_grad_fcn_kwargs': {},
                                                     'kwargs': {}
                                                     }}])
def test_no_numerical_diff_scheme__validate_wachter_loss_spec(loss_spec, predictor_type):
    with pytest.raises(CounterfactualError) as excinfo:
        _validate_wachter_loss_spec(loss_spec, predictor_type)
    assert 'Missing key \'num_grad_method\'' in str(excinfo.value)


@pytest.mark.parametrize('X', [np.random.rand(3, 4, 5)])
def test__select_features_all(X):
    mask = _select_features(X, feature_whitelist='all')
    assert_array_almost_equal(mask, np.ones(X.shape))

from collections import namedtuple
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from alibi.explainers.experimental.counterfactuals import _WachterCounterfactual

bounds = namedtuple('bounds', 'lb ub')


@pytest.mark.parametrize("lams", [np.logspace(-1, -3, num=3)])  # [1e-1, 1e-2, 1e-3]
@pytest.mark.parametrize("cf_found,expected_bounds", [(np.array([1, 1, 1]), bounds(lb=1e-1, ub=1.)),  # 10x lower bound
                                                      (np.array([0, 1, 1]), bounds(lb=1e-3, ub=1e-1)),  # standard
                                                      (np.array([0, 1, 0]), bounds(lb=1e-2, ub=1e-1)),  # impossible?
                                                      (np.array([1, 0, 0]), bounds(lb=1e-1, ub=1.))])  # 10x lower bound
def test_compute_lam_bounds(cf_found, lams, expected_bounds):
    bounds = _WachterCounterfactual.compute_lam_bounds(cf_found, lams)
    lb, ub = bounds.lb, bounds.ub
    assert_almost_equal(lb, expected_bounds.lb)
    assert_almost_equal(ub, expected_bounds.ub)

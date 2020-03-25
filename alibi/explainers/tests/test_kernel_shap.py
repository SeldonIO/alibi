# type: ignore
import itertools
import logging

import pandas
import pytest
import scipy.sparse
import shap

import numpy as np
import pandas as pd
import sklearn

from alibi.api.defaults import DEFAULT_META_SHAP, DEFAULT_DATA_SHAP
from alibi.explainers.kernel_shap import sum_categories, BACKGROUND_WARNING_THRESHOLD
from alibi.explainers.tests.utils import get_random_matrix
from alibi.tests.utils import assert_message_in_logs
from copy import copy
from itertools import chain
from numpy.testing import assert_allclose, assert_almost_equal
from shap.common import DenseData

SUPPORTED_BACKGROUND_DATA_TYPES = ['data', 'array', 'sparse', 'frame', 'series']


# Functions for data generation


def gen_group_names(n_groups):
    """
    Generate `n_groups` random names.
    """

    if n_groups == 0:
        return

    return [str(i) for i in range(n_groups)]


def random_ints_with_sum(n):
    """
    Generate positive random integers summing to `n`, sampled
    uniformly from the ordered integer partitions of `n`.
    """

    p = 0
    for _ in range(n - 1):
        p += 1
        if np.random.randint(0, 2):
            yield p
            p = 0
    yield p + 1


def gen_random_groups(n_cols):
    """
    Simulate data grouping for an array with `n_cols` columns.
    """

    if n_cols == 0:
        return

    partition_sizes = list(random_ints_with_sum(n_cols))
    groups = [list(range(0, partition_sizes[0]))]
    for size in partition_sizes[1:]:
        start = groups[-1][-1] + 1
        end = start + size
        groups.append(list(range(start, end)))

    # sanity checks
    assert sum(len(g) for g in groups) == n_cols
    for i, el in zip(range(n_cols), list(chain(*groups))):
        assert i == el

    return groups


def gen_random_weights(n_weights, seed=None):
    """
    Generate randomly an array with `n_weights` summing to 1.
    """

    np.random.seed(seed)

    if n_weights == 0:
        return

    return np.random.dirichlet(alpha=np.ones(n_weights))


def setup_groups_and_weights(dimensions, b_group_names, b_groups, b_weights, seed=None):
    """
    Generates random groups of columns, along with group names and weights, depending
    on the values of the corresponding boleans b_*.
    """

    np.random.seed(seed)

    n_features, n_samples = dimensions
    if n_samples == 0:
        n_samples += 1

    groups = gen_random_groups(n_features * int(b_groups))
    if b_group_names:
        if groups:
            group_names = gen_group_names(len(groups))
        else:
            group_names = gen_group_names(n_features)
    else:
        group_names = None
    # avoid errors when dim of background data has 1 dimension
    weights = gen_random_weights(n_samples) if b_weights else None

    return group_names, groups, weights


def get_data(kind='array', n_rows=15, n_cols=49, fnames=None, seed=None):
    """
    Generates random data with a specified type for the purposes
    of testing grouping functionality of the wrapper.
    """

    np.random.seed(seed)

    if kind not in SUPPORTED_BACKGROUND_DATA_TYPES:
        msg = "Selected data type, {}, is not an allowed type. " \
              "Allowed types are {}"
        raise ValueError(msg.format(kind, SUPPORTED_BACKGROUND_DATA_TYPES))

    X = get_random_matrix(n_rows=n_rows, n_cols=n_cols)

    if kind == 'array':
        return X
    elif kind == 'sparse':
        return scipy.sparse.csr_matrix(X)
    elif kind == 'frame' or kind == 'series':
        if not fnames:
            fnames = ['feature_{}'.format(i) for i in range(X.shape[-1])]
        if kind == 'frame':
            return pd.DataFrame(data=X, columns=fnames)
        else:
            idx = np.random.choice(np.arange(X.shape[0]))
            return pd.DataFrame(data=X, columns=fnames).iloc[idx, :]
    elif kind == 'data':
        if not fnames:
            group_names = ['feature_{}'.format(i) for i in range(X.shape[-1])]
        else:
            group_names = fnames
        return DenseData(X, group_names)
    else:
        return 0


def generate_test_data(dimensions,
                       b_group_names,
                       b_groups,
                       b_weights,
                       correct=True,
                       error_type=None,
                       data_type='',
                       dim_mismatch=3,
                       seed=None):
    """
    Generates:
        - a random dataset `data` with dim `dimensions` of type `data_type`
        - depending on the values of the `b_*` inputs groups of indices (`groups`),
        `group_names` and `weights` for each data point are also generated
        - if `correct=True`, then the inputs should not causes errors. Otherwise,
        depending on the value of `error_type`  the `data` array or the groups and
        group_names lists are specified so that specific errors or warnings are generated.
        `dim_mismatch` controls dimension mismatches
    """

    np.random.seed(seed)
    # create dimension mismatches by removing `dim_mismatch` rows/columns from the data
    if all([dim <= dim_mismatch for dim in dimensions]):
        raise ValueError(
            "dim_mismatch has to be greater than at least one dimension in order to "
            "correctly generate erroneous data!",
        )

    # generate groups setup and data according to booleans for groups and weights
    n_samples, n_features = dimensions

    group_names, groups, weights = setup_groups_and_weights(
        (n_features, n_samples),
        b_group_names,
        b_groups,
        b_weights
    )

    # we modify the data or the groups so that the input is
    # no longer correct to check warnings
    if not correct:
        # switch type of group without affecting dimensions
        if error_type == 'groups_type':
            if b_groups:
                dummy_group = [str(x) for x in range(len(groups[0]))]
                groups[0] = ''.join(dummy_group)
        # name_dim_mismatch: only specified group names (not groups)
        # but the nb. of name doesn't match any data dimension
        # groups_dim_mismatch: expected dim according to groups does
        # not match any data dimension
        elif error_type == 'name_dim_mismatch' or error_type == 'groups_dim_mismatch':
            n_features -= dim_mismatch
            if n_samples not in (0, 1):
                n_samples -= dim_mismatch
            # we don't test weights warnings
            if weights is not None:
                weights = weights[:-dim_mismatch]
        # expected dim according to grouping matches first not second dim
        # so we transpose the data
        elif error_type == 'check_transpose':
            n_features, n_samples = n_samples, n_features
        # less weights compared to data points
        elif error_type == 'weights_dim_mismatch':
            if n_samples == 1:
                n_samples += dim_mismatch
            else:
                n_samples -= dim_mismatch
        # number of groups is different to number of group names
        elif error_type == 'groups_group_names_mismatch':
            if b_groups and b_group_names:
                group_names = group_names[:-1]

    data = get_data(data_type, n_rows=n_samples, n_cols=n_features)

    return group_names, groups, weights, data


# Assertion functions


def assert_groups(background_data, group_names, groups):
    """
    Helper function to check grouping works as intended.
    """

    assert isinstance(background_data, shap.common.Data)
    assert background_data.group_names is not None

    for d, a in zip(background_data.group_names, group_names):
        assert d == a
    if groups:
        for d_group, a_group in zip(groups, background_data.groups):
            for d_idx, a_idx in zip(d_group, a_group):
                assert d_idx == a_idx
    else:
        assert len(background_data.groups) == len(group_names)


class KMeansMock:

    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)

    def _mock_kmeans(self, data, n_clusters):
        return sklearn.utils.resample(data, n_samples=n_clusters, random_state=self.seed)

    def __call__(self, background_data, n_background_samples):
        sampled = self._mock_kmeans(background_data, n_background_samples)
        group_names = [str(i) for i in range(background_data.shape[1])]

        if isinstance(background_data, pandas.DataFrame):
            group_names = background_data.columns

        return DenseData(sampled, group_names, None)


# Tests below


sum_categories_inputs = [
    (50, [3, 6, 4, 4], None),
    (50, None, [0, 6, 5, 12]),
    (100, [3, 6, 4, 4], [0, 6, 15, 22]),
    (5, [3, 2, 4], [0, 5, 9]),
    (10, [3, 3, 4], [0, 3, 6])
]


@pytest.mark.parametrize('n_feats, feat_enc_dim, start_idx', sum_categories_inputs)
def test_sum_categories(n_feats, feat_enc_dim, start_idx):
    """
    Tests if summing the columns corresponding to categorical
    variables into one variable works properly.
    """

    # create inputs to feed the function
    X = get_random_matrix(n_cols=n_feats)

    # check a value correct is raised if start indices or
    # encoding lengths are not provided
    if feat_enc_dim is None or start_idx is None:
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError
    elif len(feat_enc_dim) != len(start_idx):
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError

    # check if sum of encodings greater than num columns raises value correct
    elif sum(feat_enc_dim) > n_feats:
        with pytest.raises(ValueError) as exc_info:
            summ_X = sum_categories(X, start_idx, feat_enc_dim)
            assert exc_info.type is ValueError

    # check that if inputs are correct, we retrieve the sum in the correct col
    else:
        summ_X = sum_categories(X, start_idx, feat_enc_dim)
        assert summ_X.shape[1] == X.shape[1] - sum(feat_enc_dim) + len(feat_enc_dim)
        for i, enc_dim in enumerate(feat_enc_dim):
            # work out the index of the summed column in the returned matrix
            sum_col_idx = start_idx[i] - sum(feat_enc_dim[:i]) + len(feat_enc_dim[:i])
            diff = summ_X[:, sum_col_idx] - np.sum(X[:, start_idx[i]:start_idx[i] + feat_enc_dim[i]], axis=1)
            assert diff.sum() == 0.0


# each tuple in group_settings controls whether the
# group_names, groups or weights arguments are passed to
# KernelShap._get_data. The data is generated randomly
# as a function of `n_features` and `n_samples` by functions
# defined above

group_settings = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]
input_settings = [{'correct': True, 'error_type': None}]
data_type = copy(SUPPORTED_BACKGROUND_DATA_TYPES)
data_type.append('int')
n_classes = [(5, 'identity'), ]


# @pytest.mark.skip
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes={}'.format)
@pytest.mark.parametrize('data_dimension', ((15, 49),), ids='n_samples_feats={}'.format)
@pytest.mark.parametrize('data_type', SUPPORTED_BACKGROUND_DATA_TYPES, ids='data_type={}'.format)
@pytest.mark.parametrize('group_settings', group_settings, ids='group_names, groups, weights={}'.format)
@pytest.mark.parametrize('input_settings', input_settings, ids='input={}'.format)
def test__get_data(mock_ks_explainer, data_dimension, data_type, group_settings, input_settings):
    """
    Tests the _get_data method of the wrapper.
    """

    # generate fake inputs for testing (b_ = bool var)
    b_group_names, b_groups, b_weights = group_settings
    use_groups = b_group_names or b_groups
    group_names, groups, weights, data = generate_test_data(
        data_dimension,
        *group_settings,
        **input_settings,
        data_type=data_type,
    )
    # the algorithm would take this step in fit before calling _get_data
    if not b_group_names and b_groups:
        group_names = ['group_i'.format(i) for i in range(len(groups))]

    # initialise a KernelShap with a mock predictor
    explainer = mock_ks_explainer
    explainer.use_groups = use_groups
    explainer.summarise_background = False

    if data_type == 'int':
        with pytest.raises(TypeError) as exc_info:
            background_data = explainer._get_data(data, group_names, groups, weights)
            assert not background_data
            assert exc_info.type is TypeError
    else:
        background_data = explainer._get_data(data, group_names, groups, weights)

        # test behaviour when input is a shap.common.DenseData object
        if data_type == 'data':
            assert isinstance(background_data, shap.common.Data)
            assert background_data.group_names is not None
            if weights is not None:
                assert len(np.unique(background_data.weights)) == 1

        # test behaviour when provided data is numpy array
        if data_type == 'array':
            if use_groups:
                assert_almost_equal(background_data.weights.sum(), 1.0)
                assert_groups(background_data, group_names, groups)
            else:
                assert isinstance(background_data, np.ndarray)

        # test behaviour when provided data is sparse array
        if data_type == 'sparse':
            if use_groups:
                assert_allclose(background_data.data, data.toarray())
                assert_groups(background_data, group_names, groups)
            else:
                assert isinstance(background_data, scipy.sparse.spmatrix)

        # test behaviour when provided data is a pandas DataFrame
        if data_type == 'frame':
            if use_groups:
                expected_groups = data.columns
                assert_groups(background_data, expected_groups, groups)
            else:
                assert isinstance(background_data, pd.DataFrame)

        # test behaviour when provided data is a pandas Series
        if data_type == 'series':
            if use_groups:
                expected_groups = data.index
                assert_groups(background_data, expected_groups, groups)
            else:
                assert isinstance(background_data, pd.Series)


group_settings = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]
data_types = copy(SUPPORTED_BACKGROUND_DATA_TYPES)
input_settings = [
    {'correct': False, 'error_type': 'groups_type'},
    {'correct': False, 'error_type': 'name_dim_mismatch'},
    {'correct': False, 'error_type': 'groups_dim_mismatch'},
    {'correct': False, 'error_type': 'check_transpose'},
    {'correct': False, 'error_type': 'weights_dim_mismatch'},
    {'correct': False, 'error_type': 'groups_group_names_mismatch'},
    {'correct': True, 'error_type': None},
]
n_classes = [(5, 'identity'), ]
data_dimensions = [(BACKGROUND_WARNING_THRESHOLD + 5, 49), (55, 49), (1, 49)]
summarise_background = [True, False]


def uncollect_if_test_check_inputs(**kwargs):
    error_type = kwargs['input_settings']['error_type']
    group_settings = kwargs['group_settings']
    summarise_background = kwargs['summarise_background']
    data_dimension = kwargs['data_dimension']
    if len(data_dimension) == 2:
        n_samples, _ = data_dimension
    else:
        n_samples, _ = 1, data_dimension
    data_type = kwargs['data_type']
    b_group_names, b_groups, b_weights = group_settings

    # see def. of error_type in generate_test_data to understand
    # why these tests are skipped
    conditions = [
        error_type == 'groups_type' and not b_groups,
        error_type == 'name_dim_mismatch' and (b_groups or not b_group_names),
        error_type == 'groups_dim_mismatch' and not b_groups,
        error_type == 'check_transpose' and not b_groups,
        error_type == 'weights_dim_mismatch' and not b_weights,
        error_type == 'groups_group_names_mismatch' and (not b_groups and not b_group_names),
        error_type == 'groups_group_names_mismatch' and (not b_groups and b_group_names),
        error_type == 'groups_group_names_mismatch' and (b_groups and not b_group_names),
        data_type == 'series' and n_samples == 1,
        summarise_background and not b_weights,
    ]

    return any(conditions)


# @pytest.mark.skip
@pytest.mark.uncollect_if(func=uncollect_if_test_check_inputs)
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes={}'.format)
@pytest.mark.parametrize('data_type', data_types, ids='data_type={}'.format)
@pytest.mark.parametrize('data_dimension', data_dimensions, ids='n_feats_samples={}'.format)
@pytest.mark.parametrize('group_settings', group_settings, ids='group_names, groups, weights={}'.format)
@pytest.mark.parametrize('input_settings', input_settings, ids='input={}'.format)
@pytest.mark.parametrize('summarise_background', summarise_background, ids='summarise={}'.format)
def test__check_inputs(caplog,
                       mock_ks_explainer,
                       data_type,
                       data_dimension,
                       group_settings,
                       input_settings,
                       summarise_background,
                       ):
    """
    Tests that the _check_inputs method logs the expected warnings and info messages.
    """

    caplog.set_level(logging.INFO)

    # generate grouping inputs for testing (b_ = bool var)
    b_group_names, b_groups, b_weights = group_settings
    use_groups = b_group_names or b_groups
    if len(data_dimension) == 1:
        data_dimension = (0, data_dimension[0])
    group_names, groups, weights, data = generate_test_data(
        data_dimension,
        *group_settings,
        **input_settings,
        data_type=data_type,
    )

    _, error_type = input_settings['correct'], input_settings['error_type']

    # initialise a KernelShap with a mock predictor
    explainer = mock_ks_explainer
    explainer.use_groups = use_groups
    explainer.summarise_background = summarise_background
    explainer._check_inputs(data, group_names, groups, weights)
    records = caplog.records

    #
    grouping_errors = [
        'groups_dim_mismatch',
        'groups_type',
        'groups_group_names_mismatch',
        'name_dim_mismatch',
    ]

    # if shap.common.Data is passed, expect no warnings
    if data_type == 'data':
        if summarise_background:
            if data.data.shape[0] > BACKGROUND_WARNING_THRESHOLD:
                msg_start = 'Large datasets can cause slow runtimes for shap.'
                assert_message_in_logs(msg_start, records)
        else:
            assert not records
            assert not explainer.use_groups
    elif data.ndim == 1 or data.shape[0] == 1:  # pd.Series or single row
        if b_weights:
            assert explainer.ignore_weights
        if error_type in grouping_errors:
            assert not explainer.use_groups
        else:
            if error_type != 'check_transpose':
                if use_groups:
                    assert explainer.use_groups
            else:
                assert not explainer.use_groups
            assert not explainer.transposed
    else:
        if data.shape[0] > BACKGROUND_WARNING_THRESHOLD:
            msg_start = 'Large datasets can cause slow runtimes for shap.'
            assert_message_in_logs(msg_start, records)

        # if the group names are specified but no lists with indices for each
        # groups are passed, we inform the users that we assume len=1 groups.
        if b_group_names and not b_groups:
            msg_start = "Specified group_names but no corresponding sequence 'groups' " \
                        "with indices for each group was specified."
            assert_message_in_logs(msg_start, records)
            if error_type == 'name_dim_mismatch':
                assert not explainer.use_groups
            else:
                assert explainer.use_groups

        # if group names are not specified but groups are, we warn the user
        # and automatically create group names.
        if b_groups and not b_group_names:
            assert explainer.create_group_names

        if b_groups:
            if error_type in grouping_errors:
                assert not explainer.use_groups
            elif error_type == 'check_transpose':
                assert explainer.transposed
            else:
                assert explainer.use_groups
                assert not explainer.transposed

        if b_weights:
            if error_type == 'weights_dim_mismatch':
                assert explainer.ignore_weights
            elif error_type == 'check_transposed':
                assert not explainer.ignore_weights
            else:
                assert not explainer.ignore_weights


data_types = copy(SUPPORTED_BACKGROUND_DATA_TYPES)
n_classes = [(5, 'identity'), ]  # second element refers to the predictor link function
data_dimension = [(BACKGROUND_WARNING_THRESHOLD + 5, 49), ]
use_groups = [True, False]
categorical_names = [{}, {1: ['a', 'b', 'c']}]


# @pytest.mark.skip
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_outs, link={}'.format)
@pytest.mark.parametrize('data_type', data_types, ids='data_type={}'.format)
@pytest.mark.parametrize('data_dimension', data_dimension, ids='n_feats_samples={}'.format)
@pytest.mark.parametrize('use_groups', use_groups, ids='use_groups={}'.format)
@pytest.mark.parametrize('categorical_names', categorical_names, ids='categorical_names={}'.format)
def test__summarise_background(mock_ks_explainer, caplog, data_dimension, data_type, use_groups, categorical_names):
    caplog.set_level(logging.INFO)
    # create testing inputs
    n_samples, n_features = data_dimension
    n_bckg_samples = n_samples - 5
    background_data = get_data(data_type, n_rows=n_samples, n_cols=n_features)

    # initialise explainer
    explainer = mock_ks_explainer
    explainer.categorical_names = categorical_names
    explainer.use_groups = use_groups
    summary_data = explainer._summarise_background(background_data, n_bckg_samples)

    if data_type == 'data':
        msg = "Received option to summarise the data but the background_data object was an " \
              "instance of shap.common.Data"
        assert_message_in_logs(msg, caplog.records)
        assert type(background_data) == type(summary_data)
    else:
        if use_groups or categorical_names:
            assert type(background_data) == type(summary_data)
            if data_type == 'series':
                assert summary_data.shape == background_data.shape
            else:
                assert summary_data.shape == (n_bckg_samples, n_features)
        else:
            if data_type == 'sparse':
                assert summary_data.shape == (n_bckg_samples, n_features)
            elif data_type == 'series':
                assert summary_data.shape == background_data.shape
            else:
                assert isinstance(summary_data, shap.common.DenseData)
                assert summary_data.data.shape == (n_bckg_samples, n_features)


data_types = copy(SUPPORTED_BACKGROUND_DATA_TYPES)
data_types.remove('series')  # internal error from shap due to dimension of output being 0
group_settings = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]
# NB: input_settings used directly and not a input to generate_test_data.
# The data is generated correctly but the number of weights is modified
# based on these inputs to check warnings are raised is used summarises
# the data but doesn't pass the right number of weights

input_settings = [
    {'correct': True, 'error_type': None},
    {'correct': False, 'error_type': 'weights_dim_mismatch'},
]
data_dimensions = [(BACKGROUND_WARNING_THRESHOLD + 5, 49), (49, 49), ]
n_classes = [(5, 'identity'), (1, 'identity'), ]


def uncollect_if_test_fit(**kwargs):
    _, _, b_weights = kwargs['group_settings']
    error_type = kwargs['input_settings']['error_type']
    summarise_background = kwargs['summarise_background']

    conditions = [
        error_type == 'weights_dim_mismatch' and not b_weights,
        error_type == 'weights_dim_mismatch' and not summarise_background
    ]

    return any(conditions)


# @pytest.mark.skip
@pytest.mark.uncollect_if(func=uncollect_if_test_fit)
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes, link={}'.format)
@pytest.mark.parametrize('data_type', data_types, ids='data_type={}'.format)
@pytest.mark.parametrize('summarise_background', [True, False, 'auto'], ids='summarise={}'.format)
@pytest.mark.parametrize('data_dimension', data_dimensions, ids='n_samples_feats={}'.format)
@pytest.mark.parametrize('group_settings', group_settings, ids='group_names, groups, weights={}'.format)
@pytest.mark.parametrize('input_settings', input_settings, ids='input={}'.format)
def test_fit(caplog,
             monkeypatch,
             mock_ks_explainer,
             data_type,
             summarise_background,
             data_dimension,
             group_settings,
             input_settings):
    """
    This is an integration test where we check that the _check_inputs, _get_data and _summarise_background
    methods work well together.
    """

    caplog.set_level(logging.INFO)

    # generate grouping inputs for testing (b_ = bool var)
    b_group_names, b_groups, b_weights = group_settings
    n_samples, n_features = data_dimension
    n_background_examples = n_samples - 10
    group_names, groups, weights, data = generate_test_data(
        data_dimension,
        *group_settings,
        correct=True,
        data_type=data_type,
    )

    # modify the number of weights so that defaults are set automatically
    if not input_settings['correct']:
        weights = weights[:-3]
    # ensure we pass the right number of weights otherwise
    else:
        if b_weights:
            if summarise_background == 'auto':
                weights = weights[:BACKGROUND_WARNING_THRESHOLD]
            elif summarise_background:
                weights = weights[:n_background_examples]

    explainer = mock_ks_explainer
    # replace kmeans with a mock object so we don't run actual kmeans a zillion times
    monkeypatch.setattr(shap, "kmeans", KMeansMock())
    # check weights are not set
    explainer.fit(
        data,
        summarise_background=summarise_background,
        n_background_samples=n_background_examples,
        group_names=group_names,
        groups=groups,
        weights=weights,
    )
    records = caplog.records
    explainer = mock_ks_explainer

    n_outs = explainer.predictor.out_dim
    if n_outs == 1:
        msg = "Predictor returned a scalar value"
        assert_message_in_logs(msg, records)

    if not input_settings['correct']:
        # this only tests the case where the data is summarise
        if data_type not in ['data', 'series']:
            assert explainer.ignore_weights

        # uniform weights should be set by default
        if explainer.use_groups:
            assert len(np.unique(explainer.background_data.weights)) == 1
        else:
            if data_type != 'sparse':
                assert len(explainer.background_data.weights) != len(weights)

    else:
        if summarise_background or isinstance(summarise_background, str):
            if data_type == 'data' or data_type == 'series':
                msg_end = "No summarisation will take place!"
                assert_message_in_logs(msg_end, records)
            else:
                if explainer.use_groups:
                    background_data = explainer.background_data
                    assert isinstance(background_data, shap.common.Data)

                    # check properties of background data are correct
                    if b_group_names:
                        background_data.group_names == group_names
                    if b_groups:
                        assert background_data.groups == groups
                    if b_groups and not b_group_names:
                        if data_type not in ['frame', 'series']:
                            assert 'group' in background_data.group_names[0]
                        else:
                            assert 'feature' in background_data.group_names[0]
                    if b_weights:
                        assert len(np.unique(background_data.weights)) != 1

                    background_data = background_data.data
                else:
                    if data_type == 'sparse':
                        background_data = explainer.background_data
                        assert isinstance(background_data, type(data))
                    else:
                        assert isinstance(explainer.background_data, shap.common.Data)
                        background_data = explainer.background_data.data

                # check dimensions are reduced
                if isinstance(summarise_background, str):
                    if n_samples > BACKGROUND_WARNING_THRESHOLD:
                        assert background_data.shape[0] == BACKGROUND_WARNING_THRESHOLD
                    else:
                        assert background_data.shape[0] == data.shape[0]
                elif summarise_background:
                    assert background_data.shape[0] == n_background_examples
        else:
            background_data = explainer.background_data
            if not explainer.use_groups and data_type != 'data':
                assert background_data.shape == data.shape
            else:
                assert isinstance(background_data, shap.common.Data)
                if b_groups and not b_group_names:
                    # use columns/index for feat names for frame/series
                    # we don't check shap.commmon.Data objects
                    if data_type not in ['frame', 'series', 'data']:
                        assert 'group' in background_data.group_names[0]
                    else:
                        assert 'feature' in background_data.group_names[0]


data_types = copy(SUPPORTED_BACKGROUND_DATA_TYPES)
data_types.remove('data')
data_types.remove('series')
n_classes = [(5, 'identity'), ]
use_groups = [True, False]
summarise_result = [True, False]


# @pytest.mark.skip
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes, link={}'.format)
@pytest.mark.parametrize('use_groups', use_groups, ids='use_groups={}'.format)
@pytest.mark.parametrize('summarise_result', summarise_result, ids='summarise_result={}'.format)
@pytest.mark.parametrize('data_type', data_types, ids='data_type={}'.format)
def test_explain(monkeypatch, mock_ks_explainer, use_groups, summarise_result, data_type):
    """
    Integration tests, runs .explain method to check output dimensions are as expected.
    """
    # create fake data and records to explain
    seed = 0
    n_feats, n_samples, n_instances = 15, 20, 2
    background_data = get_data(data_type, n_rows=n_samples, n_cols=n_feats, seed=seed)
    instances = get_data(data_type, n_rows=n_instances, n_cols=n_feats, seed=seed + 1)

    # create groups
    if use_groups:
        group_names, groups, _ = setup_groups_and_weights(
            (n_feats, n_samples),
            b_group_names=True,
            b_groups=True,
            b_weights=False,
            seed=seed,
        )

    else:
        groups, group_names = None, None

    # create arrays with categorical variables start indices and encodings dim
    if summarise_result:
        if use_groups:
            cat_vars_start_idx, cat_vars_enc_dim = [], []
            start_idx = 0
            for group in groups:
                if len(group) > 1:
                    cat_vars_start_idx.append(start_idx)
                    cat_vars_enc_dim.append(len(group))
                    start_idx += len(group)
                else:
                    start_idx += 1
            if cat_vars_start_idx is None:
                grp = itertools.chain(groups[-2:])
                groups = groups[:-2]
                groups.append(grp)
                cat_vars_start_idx.append(n_feats - 2)
                cat_vars_enc_dim.append(2)
        else:
            cat_vars_start_idx, cat_vars_enc_dim = [0], [2]
    else:
        cat_vars_start_idx, cat_vars_enc_dim = None, None

    # initialise and fit explainer
    explainer = mock_ks_explainer
    monkeypatch.setattr(shap, "kmeans", KMeansMock())
    explainer.use_groups = use_groups
    explainer.fit(background_data, group_names=group_names, groups=groups)

    # explain some instances
    explanation = explainer.explain(
        instances,
        summarise_result=summarise_result,
        cat_vars_enc_dim=cat_vars_enc_dim,
        cat_vars_start_idx=cat_vars_start_idx,
        nsamples=4,
    )

    # check that explanation metadata and data keys are as expected
    assert explanation.meta.keys() == DEFAULT_META_SHAP.keys()
    assert explanation.data.keys() == DEFAULT_DATA_SHAP.keys()

    # check the output has expected shapes given the inputs
    n_outs = explainer.predictor.out_dim
    shap_values = [val for val in explanation['shap_values']]
    n_shap_values = [arr.shape[1] for arr in shap_values]
    inst_explained = [arr.shape[0] for arr in shap_values]
    assert len(set(inst_explained)) == 1
    shap_dims = set(n_shap_values)
    assert len(shap_dims) == 1
    assert inst_explained[0] == n_instances
    assert len(shap_values) == n_outs
    assert explanation.raw['raw_prediction'].shape == (n_instances, n_outs)
    assert explanation.raw['instances'].shape == (n_instances, n_feats)
    assert len(explanation.raw['prediction'].squeeze()) == n_instances

    # check dimensions of shap value arrays returned
    if use_groups:
        assert not explainer.summarise_result
        assert len(groups) in shap_dims
    else:
        if summarise_result:
            assert explainer.summarise_result
            assert n_feats - sum(cat_vars_enc_dim) + len(cat_vars_start_idx) in shap_dims
        else:
            assert n_feats in shap_dims


n_classes = [(5, 'identity'), ]
data_dimensions = [(100, 50), ]


# @pytest.mark.skip
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes, link={}'.format)
@pytest.mark.parametrize('data_dimension', data_dimensions, ids='n_samples_feats={}'.format)
def test_rank_by_importance(mock_ks_explainer, data_dimension):
    """
    Tests the feature effects ranking function.
    """

    def get_column_ranks(X, ascending=False):
        """
        Ranks the columns of X according to the average magnitude value
        and returns an array of ranking indices and a an array of
        sorted values according to the ranking.
        """

        avg_mag = np.mean(np.abs(X), axis=0)
        rank = np.argsort(avg_mag)
        if ascending:
            return rank, avg_mag[rank]
        else:
            return rank[::-1], avg_mag[rank][::-1]

    # setup explainer
    n_samples, n_features = data_dimension
    explainer = mock_ks_explainer
    explainer.feature_names = gen_group_names(n_features)

    # create inputs
    n_outs = explainer.predictor.out_dim
    shap_values = [get_random_matrix(n_rows=n_samples, n_cols=n_features) for _ in range(n_outs)]

    # compute desired values
    exp_ranked_effects_class = {}
    expected_feat_names_order = {}
    ranks_and_vals = [get_column_ranks(class_shap_vals) for class_shap_vals in shap_values]
    ranks, vals = list(zip(*ranks_and_vals))
    for i, values in enumerate(vals):
        exp_ranked_effects_class[str(i)] = vals[i]
        expected_feat_names_order[str(i)] = [explainer.feature_names[k] for k in ranks[i]]
    aggregate_shap = np.sum(shap_values, axis=0)
    exp_aggregate_rank, exp_ranked_effects_aggregate = get_column_ranks(aggregate_shap)
    exp_aggregate_names = [explainer.feature_names[k] for k in exp_aggregate_rank]

    # check results
    importances = explainer.rank_by_importance(shap_values)
    assert len(importances.keys()) == n_outs + 1
    for key in importances:
        if key != 'aggregated':
            assert_allclose(importances[key]['ranked_effect'], exp_ranked_effects_class[key])
            assert importances[key]['names'] == expected_feat_names_order[key]
        else:
            assert_allclose(importances[key]['ranked_effect'], exp_ranked_effects_aggregate)
            assert importances[key]['names'] == exp_aggregate_names


# pytest.mark.skip
@pytest.mark.parametrize('mock_ks_explainer', n_classes, indirect=True, ids='n_classes, link={}'.format)
def test_update_metadata(mock_ks_explainer):
    """
    Test that the metadata updates are correct.
    """

    explainer = mock_ks_explainer
    explainer._update_metadata({'wrong_arg': None, 'link': 'logit'}, params=True)
    explainer._update_metadata({'random_arg': 0}, params=False)
    metadata = explainer.meta

    assert 'wrong_arg' not in metadata['params']
    assert metadata['params']['link'] == 'logit'
    assert metadata['random_arg'] == 0

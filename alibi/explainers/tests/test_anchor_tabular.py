# flake8: noqa E731
from collections import OrderedDict

import numpy as np
import pytest

from collections import defaultdict
from copy import deepcopy

from alibi.explainers import DistributedAnchorTabular
from alibi.explainers.tests.utils import predict_fcn


# TODO: Test DistributedAnchorBaseBeam separately
def uncollect_if_test_explainer(**kwargs):
    """
    This function is used to skip combinations of explainers
    and classifiers that do not make sense. This is achieved
    by using the hooks in conftest.py. Such functions should
    be passed to the @pytest.mark.uncollect_if decorator as
    the func argument. They should take the same inputs as the
    test function. Because this test function is parametrized
    with a lazy fixture, in this case the arguments name change
    (ie explainer can be both at_iris_explainer or at_adult_explainer),
    the **kwargs argument is used to collect all inputs.
    """

    # both explainer and classifier fixtures are parametrized using a LazyFixture
    # object that has a name attribute, representing the name of the dataset fixture
    clf_dataset = kwargs['rf_classifier'].name
    exp_dataset = kwargs['explainer'].name
    clf_dataset = clf_dataset.split("_")[1]
    exp_dataset = exp_dataset.split("_")[1]
    if clf_dataset != exp_dataset:
        return True

@pytest.mark.uncollect_if(func=uncollect_if_test_explainer)
@pytest.mark.parametrize('n_explainer_runs', (100,), ids='n_exp_runs={}'.format)
@pytest.mark.parametrize('at_defaults', (0.9, 0.95), ids='at_defaults={}'.format, indirect=True)
@pytest.mark.parametrize('rf_classifier',
                         [pytest.lazy_fixture('get_iris_dataset'), pytest.lazy_fixture('get_adult_dataset')],
                         indirect=True,
                         ids='clf=rf_{}'.format,
                         )
@pytest.mark.parametrize('explainer',
                         [pytest.lazy_fixture('at_iris_explainer'), pytest.lazy_fixture('at_adult_explainer')],
                         ids='exp={}'.format,
                         )
def test_explainer(n_explainer_runs, at_defaults, rf_classifier, explainer):

    # fixture returns a fitted AnchorTabular explainer
    X_test, explainer, predict_fn, predict_type = explainer
    if predict_type == 'proba':
        instance_label = np.argmax(predict_fn(X_test[0, :].reshape(1, -1)), axis=1)
    else:
        instance_label = predict_fn(X_test[0, :].reshape(1, -1))[0]

    explainer.instance_label = instance_label

    explain_defaults = at_defaults
    threshold = explain_defaults['desired_confidence']
    n_covered_ex = explain_defaults['n_covered_ex']

    for _ in range(n_explainer_runs):
        explanation = explainer.explain(X_test[0], threshold=threshold, **explain_defaults)
        assert explainer.instance_label == instance_label
        assert explanation['precision'] >= threshold
        assert explanation['coverage'] >= 0.05

    sampler = explainer.samplers[0]
    assert sampler.instance_label == instance_label
    assert sampler.n_covered_ex == n_covered_ex

@pytest.mark.parametrize('ncpu', (2, 3))
@pytest.mark.parametrize('predict_type', ('proba', 'class'))
def test_distributed_anchor_tabular(ncpu, predict_type, get_iris_dataset, iris_rf_classifier):

    ray_installed = True
    try:
        import ray
    except ImportError:
        ray_installed = False
        assert True

    if ray_installed:
        import ray

        # inputs
        threshold = 0.95
        n_covered_ex = 11  # number of covered examples to return when anchor applies
        n_anchors_to_sample = 6  # for testing sampling function
        batch_size = 1000  # number of samples to draw during sampling

        # prepare the classifier and explainer
        X_test, X_train, Y_train, feature_names = get_iris_dataset
        clf = iris_rf_classifier
        predictor = predict_fcn(predict_type, clf)
        explainer = DistributedAnchorTabular(predictor, feature_names)
        explainer.fit(X_train, ncpu=ncpu)

        # select instance to be explained
        instance = X_test[0]
        if predict_type == 'proba':
            instance_label = np.argmax(predictor(instance.reshape(1, -1)), axis=1)
        else:
            instance_label = predictor(instance.reshape(1, -1))[0]

        # explain the instance and do basic checks on the lookups and instance labels used by samplers
        explanation = explainer.explain(instance, threshold=threshold, n_covered_ex=n_covered_ex)
        assert len(explainer.samplers) == ncpu
        actors = explainer.samplers
        for actor in actors:
            sampler = ray.get(actor._get_sampler.remote())
            ord_feats = sampler.ord_lookup.keys()
            cat_feats = sampler.cat_lookup.keys()
            enc_feats = sampler.enc2feat_idx.keys()
            assert (set(ord_feats | set(cat_feats))) == set(enc_feats)
            assert sampler.instance_label == instance_label
            assert sampler.n_covered_ex == n_covered_ex

        # check explanation
        assert explainer.instance_label == instance_label
        assert explanation['precision'] >= threshold
        assert explanation['coverage'] >= 0.05

        distrib_anchor_beam = explainer.mab
        assert len(distrib_anchor_beam.samplers) == ncpu

        # basic checks for DistributedAnchorBaseBeam
        anchor_features = list(enc_feats)
        anchor_max_len = len(anchor_features)
        assert distrib_anchor_beam.state['coverage_data'].shape[1] == anchor_max_len
        to_sample = []
        for _ in range(n_anchors_to_sample):
            anchor_len = np.random.randint(0, anchor_max_len)
            anchor = np.random.choice(anchor_features, anchor_len, replace=False)
            to_sample.append(tuple(anchor))
        to_sample = list(set(to_sample))
        current_state = deepcopy(distrib_anchor_beam.state)
        pos, total = distrib_anchor_beam.draw_samples(to_sample, batch_size)
        for p, t, anchor in zip(pos, total, to_sample):
            assert distrib_anchor_beam.state['t_nsamples'][anchor] == current_state['t_nsamples'][anchor] + t
            assert distrib_anchor_beam.state['t_positives'][anchor] == current_state['t_positives'][anchor] + p


@pytest.mark.parametrize('rf_classifier',
                         [pytest.lazy_fixture('get_iris_dataset')],
                         indirect=True,
                         ids='clf=rf_{}'.format,
                         )
@pytest.mark.parametrize('anchor', [(2, ), (10, ),  (11, ), (7, 10, 11), (3, 11)], ids='anchor={}'.format)
def test_iris_sampler(rf_classifier, at_iris_explainer, anchor):
    # inputs
    nb_samples = 100  # nb samples to draw when testing sampling
    # used for detailed sampler testing ...

    # fixture returns a fitted AnchorTabular explainer
    X_test, explainer, predict_fn, predict_type = at_iris_explainer

    # test sampler setup is correct
    assert len(explainer.samplers) == 1
    sampler = explainer.samplers[0]
    assert explainer.predictor(X_test[0].reshape(1, -1)).shape == (1,)
    assert sampler.train_data.shape == sampler.d_train_data.shape == (145, 4)
    assert (np.unique(sampler.d_train_data) == np.array([0., 1., 2., 3.])).all()
    assert not sampler.categorical_features
    assert len(sampler.numerical_features) == X_test.shape[1]

    if predict_type == 'proba':
        instance_label = np.argmax(predict_fn(X_test[0, :].reshape(1, -1)), axis=1)
    else:
        instance_label = predict_fn(X_test[0, :].reshape(1, -1))[0]

    explainer.instance_label = instance_label

    # test sampling function end2end
    train_data = sampler.train_data
    train_data_mean = np.mean(train_data, axis=0)
    train_data_3std = 3*np.std(train_data, axis=0)
    sampler.build_lookups(X_test[0, :])
    n_covered_ex = sampler.n_covered_ex
    cov_true, cov_false, labels, data, coverage, anchor_pos = sampler((0, anchor), nb_samples)
    assert cov_true.shape[0] <= n_covered_ex
    assert cov_false.shape[0] <= n_covered_ex
    assert len(labels) == nb_samples
    assert len(sampler.enc2feat_idx) == data.shape[1]
    assert coverage != -1

    # test lookups dictionary used for sampling
    ord_feats = sampler.ord_lookup.keys()
    cat_feats = sampler.cat_lookup.keys()
    enc_feats = sampler.enc2feat_idx.keys()
    assert (set(ord_feats | set(cat_feats))) == set(enc_feats)

    # now test perturbation method ...

    # Find out which bins can be sampled for categorical vars and check the data is sampled correctly
    allowed_bins, allowed_rows, unk_feat_vals = sampler.get_features_index(anchor)
    raw_data, disc_data, coverage = sampler.perturbation(anchor, nb_samples)
    assert not unk_feat_vals
    assert coverage != -1
    assert raw_data.shape[0] == disc_data.shape[0] == nb_samples

    uniq_feat_ids = list(OrderedDict.fromkeys([sampler.enc2feat_idx[enc_idx] for enc_idx in anchor]))
    uniq_feat_ids = [feat for feat in uniq_feat_ids if feat not in [f for f, _, _ in unk_feat_vals]]
    expected_bins = [allowed_bins[feat_id] for feat_id in uniq_feat_ids]

    for bins, feat_id in zip(expected_bins, uniq_feat_ids):
        sampled_bins_uniq = set(np.unique(disc_data[:, feat_id]))
        # check that we have replaced features properly with values from the same bin
        assert bins - sampled_bins_uniq == set()
        assert sampled_bins_uniq - bins == set()
        raw_data_mean = np.mean(raw_data, axis=0)
        # check features sampled are in the correct range
        assert (train_data_mean + train_data_3std - raw_data_mean > 0).all()
        assert (train_data_mean - train_data_3std - raw_data_mean < 0).all()


@pytest.mark.parametrize('anchor', ((2, ), (10, ),  (11, ), (7, 10, 11), (3, 11)), ids='anchor={}'.format)
@pytest.mark.parametrize('rf_classifier',
                         [pytest.lazy_fixture('get_adult_dataset')],
                         indirect=True,
                         ids='clf=rf_{}'.format,
                         )
def test_adult_sampler(anchor, rf_classifier, at_adult_explainer, get_adult_dataset):

    # inputs
    nb_samples = 100  # nb samples to draw when testing sampling
    # used for detailed sampler testing ...

    # fixture returns a fitted AnchorTabular explainer
    X_test, explainer, predict_fn, predict_type = at_adult_explainer
    data = get_adult_dataset
    category_map = data['metadata']['category_map']

    # test sampler setup is correct
    assert len(explainer.samplers) == 1
    sampler = explainer.samplers[0]
    assert explainer.predictor(X_test[0].reshape(1, -1)).shape == (1,)
    assert sampler.train_data.shape == sampler.d_train_data.shape == (30000, 12)
    assert len(sampler.categorical_features) == len(category_map.keys())
    assert len(sampler.numerical_features) == X_test.shape[1] - len(category_map.keys())

    if predict_type == 'proba':
        instance_label = np.argmax(predict_fn(X_test[0, :].reshape(1, -1)), axis=1)
    else:
        instance_label = predict_fn(X_test[0, :].reshape(1, -1))[0]

    explainer.instance_label = instance_label

    # test sampling function end2end
    train_data = sampler.train_data
    train_data_mean = np.mean(train_data, axis=0)[sampler.numerical_features]
    train_data_3std = 3*np.std(train_data, axis=0)[sampler.numerical_features]
    sampler.build_lookups(X_test[0, :])
    n_covered_ex = sampler.n_covered_ex
    cov_true, cov_false, labels, data, coverage, anchor_pos = sampler((0, anchor), nb_samples)
    assert cov_true.shape[0] <= n_covered_ex
    assert cov_false.shape[0] <= n_covered_ex
    assert len(labels) == nb_samples
    assert len(sampler.enc2feat_idx) == data.shape[1]
    assert coverage != -1

    # test lookups dictionary used for sampling
    ord_feats = sampler.ord_lookup.keys()
    cat_feats = sampler.cat_lookup.keys()
    enc_feats = sampler.enc2feat_idx.keys()
    assert (set(ord_feats | set(cat_feats))) == set(enc_feats)

    # now test perturbation method ...
    # find out which bins can be sampled for discretized vars and perturb some random data
    allowed_bins, allowed_rows, unk_feat_vals = sampler.get_features_index(anchor)
    raw_data, disc_data, coverage = sampler.perturbation(anchor, nb_samples)

    assert not unk_feat_vals
    assert coverage != -1
    assert raw_data.shape[0] == disc_data.shape[0] == nb_samples

    # find the indices of the features in the anchor
    uniq_feat_ids = list(OrderedDict.fromkeys([sampler.enc2feat_idx[enc_idx] for enc_idx in anchor]))
    uniq_feat_ids = [feat for feat in uniq_feat_ids if feat not in [f for f, _, _ in unk_feat_vals]]

    # find the mapping of feature ids to encoded feature ids
    feat2enc_idx = defaultdict(list)
    for enc_feat_idx, orig_feat_id in sampler.enc2feat_idx.items():
        feat2enc_idx[orig_feat_id].append(enc_feat_idx)

    # find the expected values the sampled features should have
    expected_values = []
    for feat_id in uniq_feat_ids:
        if feat_id in sampler.categorical_features:
            enc_idx = feat2enc_idx[feat_id][0]
            val = sampler.cat_lookup[enc_idx]
            expected_values.append({val})
        else:
            expected_values.append(allowed_bins[feat_id])

    # check that we have replaced features properly with values from the same bin
    # or with the correct value
    for bins, feat_id in zip(expected_values, uniq_feat_ids):
        sampled_bins_uniq = set(np.unique(disc_data[:, feat_id]))

        assert bins - sampled_bins_uniq == set()
        assert sampled_bins_uniq - bins == set()
        raw_data_mean = np.mean(raw_data, axis=0)[sampler.numerical_features]

        # check features sampled are in a sensible range for numerical features
        assert (train_data_mean + train_data_3std - raw_data_mean > 0).all()
        assert (train_data_mean - train_data_3std - raw_data_mean < 0).all()

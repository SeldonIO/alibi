# flake8: noqa E731

import numpy as np
import pytest

from copy import deepcopy

from alibi.explainers import DistributedAnchorTabular
from alibi.explainers.tests.utils import predict_fcn


# TODO: Test DistributedAnchorBaseBeam separately

@pytest.mark.parametrize('at_iris_explainer', ('proba', 'class'), indirect=True)
@pytest.mark.parametrize('at_defaults', (0.9, 0.95), indirect=True)
def test_iris(get_iris_dataset, iris_rf_classifier, at_defaults, at_iris_explainer):

    # inputs
    nb_samples = 15  # nb samples to draw when testing sampling

    # fixture returns a fitted AnchorTabular explainer
    X_test, explainer, predict_fn, predict_type = at_iris_explainer

    # test fitting
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

    # test sampling function
    sampler.build_lookups(X_test[0, :])
    anchor = tuple(sampler.enc2feat_idx.keys())
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

    explain_defaults = at_defaults
    threshold = explain_defaults['desired_confidence']
    n_covered_ex = explain_defaults['n_covered_ex']
    explanation = explainer.explain(X_test[0], threshold=threshold, **explain_defaults)
    assert explainer.instance_label == instance_label
    assert sampler.instance_label == instance_label
    assert sampler.n_covered_ex == n_covered_ex
    assert explanation['precision'] >= threshold
    assert explanation['coverage'] >= 0.05


@pytest.mark.parametrize('predict_type', ('proba', 'class'))
@pytest.mark.parametrize('ncpu', (2, 3))
def test_distributed_anchor_tabular(ncpu, predict_type, get_iris_dataset, iris_rf_classifier):

    ray_installed = True
    try:
        import ray
    except AttributeError:
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

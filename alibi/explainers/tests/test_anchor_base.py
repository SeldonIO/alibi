import pytest

import numpy as np

from copy import deepcopy


@pytest.mark.parametrize('rf_classifier',
                         [pytest.lazy_fixture('get_iris_dataset')],
                         indirect=True,
                         ids='clf=rf_{}'.format,
                         )
@pytest.mark.parametrize('at_defaults', (0.9, 0.95), indirect=True)
def test_anchor_base_beam(rf_classifier, at_defaults, at_iris_explainer):

    # inputs
    n_anchors_to_sample = 6
    coverage_samples = 500
    dummy_coverage = - 0.55  # used to test coverage updates on sampling

    X_test, explainer, predict_fn, predict_type = at_iris_explainer
    explain_defaults = at_defaults
    threshold = explain_defaults['desired_confidence']
    explanation = explainer.explain(X_test[0], threshold=threshold, **explain_defaults)
    anchor_beam = explainer.mab

    assert anchor_beam.state['coverage_data'].shape[0] == explain_defaults['coverage_samples']

    # Test draw_samples method
    anchor_features = list(explainer.samplers[0].enc2feat_idx.keys())
    anchor_max_len = len(anchor_features)
    assert anchor_beam.state['coverage_data'].shape[1] == anchor_max_len
    to_sample = []
    for _ in range(n_anchors_to_sample):
        anchor_len = np.random.randint(0, anchor_max_len)
        anchor = np.random.choice(anchor_features, anchor_len, replace=False)
        to_sample.append(tuple(anchor))
    to_sample = list(set(to_sample))
    current_state = deepcopy(anchor_beam.state)
    for anchor in to_sample:
        if anchor not in current_state['t_nsamples']:
            anchor_beam.state['t_coverage'][anchor] = dummy_coverage
    pos, total = anchor_beam.draw_samples(to_sample, explain_defaults['batch_size'])
    for p, t, anchor in zip(pos, total, to_sample):
        assert anchor_beam.state['t_nsamples'][anchor] == current_state['t_nsamples'][anchor] + t
        assert anchor_beam.state['t_positives'][anchor] == current_state['t_positives'][anchor] + p
        if anchor:  # empty anchor has dummy coverage
            assert anchor_beam.state['t_coverage'][anchor] != dummy_coverage

    # testing resampling works
    # by sampling all features, we are guaranteed that partial anchors might not exist
    feat_set = tuple(range(anchor_max_len))
    pos, total = anchor_beam.draw_samples([feat_set], explain_defaults['batch_size'])
    assert 'placeholder' not in explanation['raw']['examples']
    assert -1 not in explanation['raw']['coverage']

    # test anchor construction
    len_1_anchors_set = anchor_beam.propose_anchors([])
    assert len(len_1_anchors_set) == anchor_max_len
    len_2_anchors_set = anchor_beam.propose_anchors(len_1_anchors_set)
    assert len(len_2_anchors_set) == anchor_max_len*(anchor_max_len - 1)/2

    # test coverage data sampling
    cov_data = anchor_beam._get_coverage_samples(coverage_samples)
    assert cov_data.shape[0] == coverage_samples

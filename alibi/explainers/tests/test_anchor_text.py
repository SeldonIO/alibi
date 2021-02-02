# flake8: noqa E731
import pytest
from pytest_lazyfixture import lazy_fixture
import spacy

import numpy as np

from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR
from alibi.explainers import AnchorText
from alibi.explainers.anchor_text import Neighbors, _load_spacy_lexeme_prob
from alibi.explainers.tests.utils import predict_fcn
from alibi.utils.download import spacy_model

# load spaCy model
model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)


@pytest.mark.parametrize('text, n_punctuation_marks, n_unique_words',
                         [('This is a good book.', 1, 6),
                          ('I, for one, hate it.', 3, 7)])
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize("predict_type, anchor, use_similarity_proba, use_unk, threshold", [
    ('proba', (), False, True, 0.95),
    ('class', (), False, True, 0.95),
    ('class', (), False, True, 0.9),
    ('class', (), True, False, 0.95),
    ('class', (3,), True, False, 0.95),
])
def test_anchor_text(lr_classifier, text, n_punctuation_marks, n_unique_words,
                     predict_type, anchor, use_similarity_proba, use_unk, threshold):
    # test parameters
    num_samples = 100
    sample_proba = .5
    top_n = 500
    temperature = 1.
    n_covered_ex = 5  # number of examples where the anchor applies to be returned

    # fit and initialise predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn(predict_type, clf, preproc=preprocessor)

    # test explainer initialization
    explainer = AnchorText(nlp, predictor)
    assert explainer.predictor(['book']).shape == (1,)

    # setup explainer
    perturb_opts = {
        'use_similarity_proba': use_similarity_proba,
        'sample_proba': sample_proba,
        'temperature': temperature,
    }
    explainer.n_covered_ex = n_covered_ex
    explainer.set_words_and_pos(text)
    explainer.set_sampler_perturbation(use_unk, perturb_opts, top_n)
    explainer.set_data_type(use_unk)
    if predict_type == 'proba':
        label = np.argmax(predictor([text])[0])
    elif predict_type == 'class':
        label = predictor([text])[0]
    explainer.instance_label = label

    assert isinstance(explainer.dtype, str)
    assert len(explainer.punctuation) == n_punctuation_marks
    assert len(explainer.words) == len(explainer.positions)

    # test sampler
    cov_true, cov_false, labels, data, coverage, _ = explainer.sampler((0, anchor), num_samples)
    if not anchor:
        assert coverage == -1
    if use_similarity_proba and len(anchor) > 0:  # check that words in present are in the proposed anchor
        assert len(anchor) * data.shape[0] == data[:, anchor].sum()

    if use_unk:
        # get list of unique words
        all_words = explainer.words
        # unique words = words in text + UNK
        assert len(np.unique(all_words)) == n_unique_words

    # test explanation
    explanation = explainer.explain(
        text,
        use_unk=use_unk,
        threshold=threshold,
        use_similarity_proba=use_similarity_proba,
    )
    assert explanation.precision >= threshold
    assert explanation.raw['prediction'].item() == label
    assert explanation.meta.keys() == DEFAULT_META_ANCHOR.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ANCHOR.keys()

    # check if sampled sentences are not cut short
    keys = ['covered_true', 'covered_false']
    for i in range(len(explanation.raw['feature'])):
        example_dict = explanation.raw['examples'][i]
        for k in keys:
            for example in example_dict[k]:
                # check that we have perturbed the sentences
                if use_unk:
                    assert 'UNK' in example or example.replace(' ', '') == text.replace(' ', '')
                else:
                    assert 'UNK' not in example
                assert example[-1] in ['.', 'K']


def test_neighbors():
    # test inputs
    w_prob = -15.
    tag = 'NN'
    top_n = 10

    neighbor = Neighbors(_load_spacy_lexeme_prob(nlp), w_prob=w_prob)
    n = neighbor.neighbors('book', tag, top_n)
    # The word itself is excluded from the array with similar words
    assert 'book' not in n['words']
    assert len(n['words']) == top_n
    assert len(n['words']) == len(n['similarities'])
    # similarity score list needs to be descending
    similarity_score = n['similarities']
    assert np.isclose((np.sort(similarity_score)[::-1] - similarity_score).sum(), 0.)

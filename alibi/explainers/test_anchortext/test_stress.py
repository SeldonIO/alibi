import pytest
from pytest_lazyfixture import lazy_fixture

import string
import numpy as np

from alibi.explainers import AnchorText
from alibi.explainers.anchor_text import LanguageModelSampler, DEFAULT_SAMPLING_LANGUAGE_MODEL
from alibi.explainers.tests.utils import predict_fcn


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize('punctuation', ['', string.punctuation])
@pytest.mark.parametrize('stopwords', [[], ['a', 'the', 'but', 'this', 'there', 'those', 'an']])
def test_precision(lang_model, lr_classifier, movie_sentiment_data, punctuation, stopwords):
    """
    Checks if the anchor precision exceeds the threshold
    """
    # unpack test data
    X_test = movie_sentiment_data['X_test']

    # select 10 examples
    n = 1
    np.random.seed(0)
    idx = np.random.choice(len(X_test), size=n, replace=False)

    # update test data
    X_test = [X_test[i] for i in idx]

    # fit and initialize predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn('class', clf, preproc=preprocessor)

    # setup perturbation options
    perturb_opts = {
        "filling_method": "parallel",
        "sample_proba": 0.5,
        "temperature": 1.0,
        "top_n": 100,
        "frac_mask_templates": 0.1,
        "stopwords": stopwords,
        "punctuation": punctuation,
    }

    # initialize exaplainer
    explainer = AnchorText(predictor=predictor, sampling_method=AnchorText.SAMPLING_LANGUAGE_MODEL,
                           language_model=lang_model, **perturb_opts)

    for i in range(n):
        text = X_test[i]

        # compute explanation
        threshold = 0.95
        explanation = explainer.explain(text, threshold=threshold)

        # check precision to be greater than the threshold
        assert explanation.precision >= threshold


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('punctuation', ['', string.punctuation])
@pytest.mark.parametrize('stopwords', [[], ['a', 'the', 'but', 'this', 'there', 'those', 'an']])
@pytest.mark.parametrize('filling_method', ['parallel'])
@pytest.mark.parametrize('sample_proba', [0.5, 0.6, 0.7])
def test_stopwords_punctuation(lang_model, punctuation, stopwords, filling_method,
                               movie_sentiment_data, sample_proba, nlp):
    """
    Checks if the sampling procedure affect the stopwords and the
    punctuation when is not supposed to.
    """
    # unpack test data
    X_test = movie_sentiment_data['X_test']

    # select 10 sampleds
    n = 10 
    np.random.seed(0)
    idx = np.random.choice(len(X_test), size=n, replace=False)

    # update test dat
    X_test = [X_test[i] for i in idx]
    assert len(X_test) == n

    # define perturb opts
    perturb_opts = {
        "sample_proba": sample_proba,
        "punctuation": punctuation,
        "stopwords": stopwords,
        "frac_mask_templates": 0.1,
    }

    # initialize sampler
    sampler = LanguageModelSampler(model=lang_model, perturb_opts=perturb_opts)

    for i in range(n):
        text = X_test[i]
        
        # process test
        processed = nlp(text)
        words = [w.text.lower() for w in processed]
        words = {w: words.count(w) for w in words}

        # set sampler perturb opts
        sampler.set_text(text)

        # get masks samples
        raw, data = sampler.create_mask((), num_samples=10, filling_method=filling_method, **perturb_opts)
        
        for j in range(len(raw)):
            mask_counts = str(raw[j]).count(lang_model.mask)
            raw[j] = str(raw[j]).replace(lang_model.mask, '', mask_counts)

            preprocessed = nlp(str(raw[j]))
            words_masks = [w.text.lower() for w in preprocessed]
            words_masks = {w: words_masks.count(w) for w in words_masks}

            # check if the stopwords were perturb and they were not supposed to
            for sw in stopwords:
                if sw in words:
                    assert words[sw] == words_masks[sw]
            
            # check if the punctuation was petrub and it was not supposed to
            for p in punctuation:
                if (p in text) and (p != '\''):  # ' is a tricky one as in words don't for which don' is a token
                    assert text.count(p) == str(raw[j]).count(p)


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('head_gt, num_tokens',
                         [
                             ('word1', 2),
                             ('word1 word2', 3),
                             ('word1 word2 word3', 4),
                             ('word1 word2 word3 word4', 5),
                             ('word1 word2 word3 word4 word5', 6)
                         ])
def test_split(lang_model, head_gt, num_tokens):
    """
    Check if the split head-tail is correctly performed
    """
    # define a very long word as tail
    tail = 'Pneumonoultramicroscopicsilicovolcanoconiosis ' * 100
    text = head_gt + ' ' + tail

    # split head tail
    head, tail, head_tokens, tail_tokens = lang_model.head_tail_split(text)

    # get the unique words in head
    unique_words = set(head.strip().split(' '))
    assert len(unique_words) == num_tokens


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('num_tokens', [30, 50, 100])
@pytest.mark.parametrize('sample_proba', [0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
@pytest.mark.parametrize('filling_method', ['parallel', 'autoregressive'])
def test_mask(lang_model, num_tokens, sample_proba, filling_method):
    """
    Tests the mean number of masked tokens match the expected one.
    """
    # define text
    text = 'word ' * num_tokens

    # define perturbation options
    perturb_opts = {
        "sample_proba": sample_proba,
        "punctuation": '',
        "stopwords": [],
        "filling_method": filling_method,
    }

    # define sampler
    sampler = LanguageModelSampler(model=lang_model, perturb_opts=perturb_opts)
    sampler.set_text(text)

    # create a bunch of masks
    raw, data = sampler.create_mask((), 10000, sample_proba, frac_mask_templates=1.0)

    # hope that law of large number holds
    empirical_mean1 = np.mean(np.sum(data == 0, axis=1))
    theoretical_mean = sample_proba * len(sampler.ids_sample)

    # compute number of mask tokens in the strings
    mask = lang_model.mask.replace(lang_model.SUBWORD_PREFIX, '')
    empirical_mean2 = np.mean([str(x).count(mask) for x in raw])
    assert np.abs(empirical_mean1 - theoretical_mean) < 1.0
    assert empirical_mean1 == empirical_mean2


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('filling_method', ['parallel'])
@pytest.mark.parametrize('punctuation', [string.punctuation])
@pytest.mark.parametrize('sample_proba', [0.5, 0.6, 0.7])
def test_sample_punctuation(lang_model, punctuation, filling_method, movie_sentiment_data, sample_proba):
    """
    Check if the punctuation is not sampled when flag is set.
    """
    # unpack test data
    X_test = movie_sentiment_data['X_test']

    # select 10 samples
    n = 10
    np.random.seed(0)
    idx = np.random.choice(len(X_test), size=n, replace=False)

    # update test dat
    X_test = [X_test[i] for i in idx]
    assert len(X_test) == n

    # define perturb opts
    perturb_opts = {
        "sample_proba": sample_proba,
        "punctuation": punctuation,
        "stopwords": [],
        "filling_method": filling_method,
        "sample_punctuation": False
    }

    # initalize sampler
    sampler = LanguageModelSampler(model=lang_model, perturb_opts=perturb_opts)

    for i in range(n):
        text = X_test[i]
        text = ''.join([chr for chr in text if chr not in string.punctuation])

        # set sampler perturb opts
        sampler.set_text(text)

        # get masks samples
        raw, data = sampler.perturb_sentence((), num_samples=10)

        for j in range(len(raw)):
            mask_counts = str(raw[j]).count(lang_model.mask)
            raw[j] = str(raw[j]).replace(lang_model.mask, '', mask_counts)

            for p in punctuation:
                assert p not in raw[j]

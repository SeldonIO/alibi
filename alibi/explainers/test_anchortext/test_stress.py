import pytest
from pytest_lazyfixture import lazy_fixture

import string
import numpy as np

from alibi.explainers import AnchorText
from alibi.explainers.anchor_text import LanguageModelSampler
from alibi.explainers.tests.utils import predict_fcn

@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased'], indirect=True)
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize('perturb_punctuation', [True, False])
@pytest.mark.parametrize('punctuation', string.punctuation)
@pytest.mark.parametrize('perturb_stopwords', [True, False])
@pytest.mark.parametrize('stopwords', [['a', 'the', 'but', 'this', 'there', 'those', 'an']])
def test_precision(lang_model, lr_classifier, movie_sentiment_data, perturb_punctuation, punctuation,
        perturb_stopwords, stopwords):
    # unpack test data
    X_test = movie_sentiment_data['X_test']

    # select 100 examples
    n = 1
    np.random.seed(0)
    idx = np.random.choice(len(X_test), size=n, replace=False)

    # update test data
    X_test = [X_test[i] for i in idx]

    # fit and initialize predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn('class', clf, preproc=preprocessor)

    # initialize exapliner
    explainer = AnchorText(language_model=lang_model, predictor=predictor)

    # setup perturbation options
    perturb_opts = {
        "sampling_method": AnchorText.SAMPLING_LANGUAGE_MODEL,
        "filling_method": "parallel",
        "sample_proba": 0.5,
        "temperature": 1.0,
        "top_n": 100,
        "threshold": 0.95,
        "perturb_punctuation": perturb_punctuation,
        "perturb_stopwords": perturb_stopwords,
        "stopwords": stopwords,
        "punctuation": punctuation,
    }
    
    for i in range(n):
        text = X_test[i]
        
        # compute explanation
        explanation = explainer.explain(
            text,
            **perturb_opts
        )
        
        # check precision to be greater than the threshold
        assert explanation.precision >= perturb_opts['threshold']


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased'], indirect=True)
@pytest.mark.parametrize('perturb_punctuation', [True, False])
@pytest.mark.parametrize('punctuation', [string.punctuation])
@pytest.mark.parametrize('perturb_stopwords', [True, False])
@pytest.mark.parametrize('stopwords', [['a', 'the', 'but', 'this', 'there', 'those', 'an']])
@pytest.mark.parametrize('filling_method', ['parallel'])
@pytest.mark.parametrize('sample_proba', [0.5, 0.6, 0.7])
def test_sample_ids(lang_model, perturb_punctuation, punctuation, perturb_stopwords, stopwords,
        filling_method, movie_sentiment_data, sample_proba, nlp):
    # unpack test data
    X_test = movie_sentiment_data['X_test']

    # select 100 sampleds
    n = 10 
    np.random.seed(0)
    idx = np.random.choice(len(X_test), size=n, replace=False)

    # update test dat
    X_test = [X_test[i] for i in idx]
    assert len(X_test) == n

    # initalize sampler
    sampler = LanguageModelSampler(model=lang_model)

    # define perturb opts
    perturb_opts = {
        "sample_proba": sample_proba,
        "perturb_punctuation": perturb_punctuation,
        "perturb_stopwords": perturb_stopwords,
        "punctuation": punctuation,
        "stopwords": stopwords,
    }

    for i in range(n):
        text = X_test[i]
        
        # process test
        processed = nlp(text)
        words = set([w.text.lower() for w in processed])

        # set sampler perturb opts
        sampler.set_params(text, perturb_opts)

        # get masks samples
        data, raw = sampler.create_mask((), num_samples=10, filling_method=filling_method, **perturb_opts)
        
        for j in range(len(raw)):
            if (not perturb_stopwords) or (not perturb_punctuation):
                preprocessed = nlp(str(raw[j]))
                words_masks = set([w.text.lower() for w in preprocessed])

            # check if the stopwords were perturb and they were not supposed to
            if not perturb_stopwords:
                for sw in stopwords:
                    if sw in words:
                        assert sw in words_masks
            
            # check if the punctuation was petrub and it was not supposed to
            if not perturb_punctuation:
                for p in punctuation:
                    if p in words:
                        assert p in words_masks

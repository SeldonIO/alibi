import pytest
from pytest_lazyfixture import lazy_fixture

import string
import numpy as np

from alibi.explainers import AnchorText, LanguageModelSampler
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
@pytest.mark.parametrize('punctuation', string.punctuation)
@pytest.mark.parametrize('perturb_stopwords', [True, False])
@pytest.mark.parametrize('stopwords', [['a', 'the', 'but', 'this', 'there', 'those', 'an']])
def test_sample_ids():
    pass

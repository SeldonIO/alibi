import pytest
import spacy

from sklearn.linear_model import LogisticRegression

from alibi.utils.lang_model import *
from alibi_testing.data import get_movie_sentiment_data
from alibi.utils.download import spacy_model


@pytest.fixture(scope='module')
def lang_model(request):
    model_name = request.param

    if model_name == "RobertaBase":
        return RobertaBase()

    if model_name == "BertBaseUncased":
        return BertBaseUncased()

    if model_name == 'DistilbertBaseUncased':
        return DistilbertBaseUncased()

    return None


@pytest.fixture(scope='module')
def movie_sentiment_data():
    return get_movie_sentiment_data()


@pytest.fixture(scope='module')
def lr_classifier(request):
    is_preprocessor = False
    preprocessor = False
    # see test_anchor_text for an example on how this
    # fixture can be parametrized
    data = request.param
    if data['preprocessor']:
        is_preprocessor = True
        preprocessor = data['preprocessor']

    clf = LogisticRegression()

    if is_preprocessor:
        clf.fit(preprocessor.transform(data['X_train']), data['y_train'])
    else:
        clf.fit(data['X_train'], data['y_train'])

    return clf, preprocessor


@pytest.fixture(scope='module')
def nlp():
    model = 'en_core_web_md'
    spacy_model(model=model)
    nlp = spacy.load(model)
    return nlp
# flake8: noqa E731

from collections import Counter
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import spacy
from alibi.explainers import AnchorText
from alibi.explainers.anchor_text import Neighbors
from alibi.datasets import fetch_movie_sentiment
from alibi.utils.download import spacy_model

# load spaCy model
model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)


def test_neighbors():
    # test Neighbors class
    neighbor = Neighbors(nlp, w_prob=-20)
    n = neighbor.neighbors('book')

    # most similar word must be the word itself with a similarity score of 1
    assert n[0][0].orth_ == 'book'
    assert n[0][1] == 1.

    # similarity score list needs to be descending
    similarity_score = [x[1] for x in n]
    assert sorted(similarity_score, reverse=True) == similarity_score


@pytest.mark.parametrize("predict_type,present,use_similarity_proba,use_unk,threshold", [
    ('class', [], False, True, 0.95),
    ('proba', [], False, True, 0.95),
    ('class', [], False, True, 0.9),
    ('class', [], True, False, 0.95),
    ('class', [3], True, False, 0.95)
])
def test_anchor_text(predict_type, present, use_similarity_proba, use_unk, threshold):
    # load data and create train and test sets
    movies = fetch_movie_sentiment()
    data = movies.data
    labels = movies.target
    train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=0)
    train_labels = np.array(train_labels)

    # apply CountVectorizer
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)

    # train Logistic Regression model
    clf = LogisticRegression()
    clf.fit(vectorizer.transform(train), train_labels)

    # define predict function
    if predict_type == 'proba':
        predict_fn = lambda x: clf.predict_proba(vectorizer.transform(x))
    elif predict_type == 'class':
        predict_fn = lambda x: clf.predict(vectorizer.transform(x))

    # test explainer initialization
    explainer = AnchorText(nlp, predict_fn)
    assert explainer.predict_fn(['book']).shape == (1,)

    # test sampling function
    text = 'This is a good book .'
    num_samples = 100
    sample_proba = .5
    top_n = 500
    words, positions, sample_fn = explainer.get_sample_fn(text, use_similarity_proba=use_similarity_proba,
                                                          use_unk=use_unk, sample_proba=sample_proba, top_n=top_n)
    raw_data, data, labels = sample_fn(present, num_samples)

    if use_similarity_proba and len(present) > 0:  # check that words in present are in the proposed anchor
        assert len(present) * data.shape[0] == data[:, present].sum()

    if use_unk:
        # get list of unique words
        all_words = []
        for i in range(raw_data.shape[0]):
            all_words.append(raw_data[i][0].split())
        all_words = [word for word_list in all_words for word in word_list]

        # unique words = words in text + UNK
        assert len(np.unique(all_words)) == len(text.split()) + 1

        # check nb of UNKs
        assert data.shape[0] * data.shape[1] - data.sum() == Counter(all_words)['UNK']

    # test explanation
    explanation = explainer.explain(text, threshold=threshold, use_proba=use_similarity_proba, use_unk=use_unk)
    assert explanation[0]['precision'] >= threshold
    # check if sampled sentences are not cut short
    keys = ['covered', 'covered_true', 'covered_false']
    for i in range(len(explanation[0]['raw']['feature'])):
        example_dict = explanation[0]['raw']['examples'][i]
        for k in keys:
            for example in example_dict[k]:
                assert example[0][-1] in ['.', 'K']

    # test serialization
    explanation.serialize()

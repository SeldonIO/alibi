import pytest
from pytest_lazyfixture import lazy_fixture

import string
import numpy as np

from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR
from alibi.explainers import AnchorText
from alibi.explainers.anchor_text import Neighbors, _load_spacy_lexeme_prob, LanguageModelSampler
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

    sampling_strategy = kwargs['sampling_strategy']
    lang_model = kwargs['lang_model']

    cond1 = (sampling_strategy != 'language_model') and (lang_model != '')
    cond2 = (sampling_strategy == 'language_model') and (lang_model == '')
    return any([cond1, cond2])


@pytest.mark.uncollect_if(func=uncollect_if_test_explainer)
@pytest.mark.parametrize('text, n_punctuation_marks, n_unique_words',
                         [('This is a good book.', 1, 6),
                          ('I, for one, hate it.', 3, 7)])
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize('predict_type, anchor, use_proba, sampling_strategy, filling, threshold',
                         [('proba', (), False, 'unknown', None, 0.95),
                          ('proba', (), False, 'similarity', None, 0.95),
                          ('proba', (), False, 'language_model', 'parallel', 0.95),
                          ('class', (), False, 'unknown', None, 0.90),
                          ('class', (), True, 'similarity', None, 0.90),
                          ('class', (), False, 'language_model', 'parallel', 0.90),
                          ('class', (3,), False, 'unknown', None, 0.95),
                          ('class', (3,), True, 'similarity', None, 0.95),
                          ('class', (3,), False, 'language_model', 'parallel', 0.95)])
@pytest.mark.parametrize('lang_model', ["", "RobertaBase", "BertBaseUncased", "DistilbertBaseUncased"], indirect=True)
def test_explainer(text, n_punctuation_marks, n_unique_words, lr_classifier, predict_type, anchor,
                   use_proba, sampling_strategy, filling, threshold, lang_model, nlp):
    # test parameters
    sample_proba = 0.5          # probability of masking a word
    top_n = 500                 # use top 500 words
    temperature = 1             # temperature coeff
    num_samples = 100           # number of sampled
    frac_mask_templates = 0.1   # fraction of masking templates
    n_covered_ex = 5            # number of examples where the anchor applies to be returned

    # fit and initialize predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn(predict_type, clf, preproc=preprocessor)

    # setup explainer
    perturb_opts = {
        'filling': filling,
        'sample_proba': sample_proba,
        'temperature': temperature,
        'top_n': top_n,
        "frac_mask_templates": frac_mask_templates,
        "use_proba": use_proba,
    }

    # test explainer initialization
    explainer = AnchorText(predictor=predictor, sampling_strategy=sampling_strategy,
                           nlp=nlp, language_model=lang_model, **perturb_opts)
    assert explainer.predictor(['book']).shape == (1, )

    # set sampler
    explainer.n_covered_ex = n_covered_ex
    explainer.perturbation.set_text(text=text)

    # set instance label
    label = np.argmax(predictor([text])[0]) if predict_type == 'proba' else predictor([text])[0]
    explainer.instance_label = label

    # check punctuation and words for `unknown` and `similarity` sampling
    if sampling_strategy in [AnchorText.SAMPLING_UNKNOWN, AnchorText.SAMPLING_SIMILARITY]:
        assert len(explainer.perturbation.punctuation) == n_punctuation_marks
        assert len(explainer.perturbation.words) == len(explainer.perturbation.positions)
    else:
        # do something similar for the transformers. this simplified verison
        # works because there are not multiple consecutive punctuations
        tokens = lang_model.tokenizer.tokenize(text)
        punctuation = []

        for token in tokens:
            if lang_model.is_punctuation(token, string.punctuation):
                punctuation.append(token.strip())

        # select all words without punctuation
        words = []

        for i, token in enumerate(tokens):
            if (not lang_model.is_subword_prefix(token)) and \
                    (not lang_model.is_punctuation(token, string.punctuation)):
                word = lang_model.select_word(tokens, i, string.punctuation)
                words.append(word.strip())

        # set with all unique words including punctuation
        unique_words = set(words) | set(punctuation)
        assert len(punctuation) == n_punctuation_marks
        assert len(unique_words) >= n_unique_words - 1  # (because of Roberta first token)

    # test sampler
    cov_true, cov_false, labels, data, coverage, _ = explainer.sampler((0, anchor), num_samples)
    if not anchor:
        assert coverage == -1

    # check that words in present are in the proposed anchor
    if (sampling_strategy in [AnchorText.SAMPLING_SIMILARITY]) and len(anchor) > 0:
        assert len(anchor) * data.shape[0] == data[:, anchor].sum()

    if sampling_strategy == AnchorText.SAMPLING_UNKNOWN:
        all_words = explainer.perturbation.words
        assert len(np.unique(all_words)) == n_unique_words

    # test explanation
    explanation = explainer.explain(text, threshold=threshold, coverage_samples=100)
    assert explanation.precision >= threshold
    assert explanation.raw['prediction'].item() == label
    assert explanation.meta.keys() == DEFAULT_META_ANCHOR.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ANCHOR.keys()


def test_neighbors(nlp):
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


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('text, min_num',
                         [("This is ... a sentence, with a long ?!?, lot of punctuation; test this.", 5)])
def test_lm_punctuation(text, min_num, lang_model):
    """
    Checks if the language model can identify punctuation correctly
    """
    head, tail, head_tokens, tail_tokens = lang_model.head_tail_split(text)

    # assert head
    assert len(head) > 0
    assert len(tail) == 0
    assert len(head_tokens) > 0
    assert len(tail_tokens) == 0

    # select tokens that are punctuation
    buff = []

    for token in head_tokens:
        if lang_model.is_punctuation(token, string.punctuation):
            buff.append(token.replace(lang_model.SUBWORD_PREFIX, '').strip())

    # make sure that it found something
    assert len(buff) >= min_num

    # check if all tokens selected contain only punctuation
    for token in buff:
        assert all([(c in string.punctuation) for c in token])


@pytest.mark.parametrize('text, stopwords',
                         [('Test the following stopwords: this, but, a, the, verylongword',
                          ['this', 'but', 'a', 'the', 'verylongword'])])
@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
def test_lm_stopwords(text, stopwords, lang_model):
    """
    Checks if the language model can identify stopwords correctly
    """
    head, tail, head_tokens, tail_tokens = lang_model.head_tail_split(text)

    # head assertions
    assert len(head) > 0
    assert len(head_tokens) > 0

    # tail assertions
    assert len(tail) == 0
    assert len(tail_tokens) == 0

    # convert stopwords to lowercase
    stopwords = [s.strip().lower() for s in stopwords]

    # find all the subwords in the sentence
    found_stopwords = []

    for i in range(len(head_tokens)):
        if lang_model.is_stop_word(head_tokens, i, string.punctuation, stopwords):
            word = lang_model.select_word(head_tokens, i, string.punctuation)
            found_stopwords.append(word.strip())

    # transform found_stopwords to lowercase
    found_stopwords = set([s.strip().lower() for s in found_stopwords])

    # check if the found subwords and the expected one are the same
    assert set(found_stopwords) == set(stopwords)


@pytest.mark.skip("This can take a while ...")
@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize('punctuation', ['', string.punctuation])
@pytest.mark.parametrize('stopwords', [[], ['a', 'the', 'but', 'this', 'there', 'those', 'an']])
def test_lm_precision(lang_model, lr_classifier, movie_sentiment_data, punctuation, stopwords):
    """
    Checks if the anchor precision exceeds the threshold
    """
    # unpack test data
    X_test = movie_sentiment_data['X_test']
    short_examples = [i for i in range(len(X_test)) if len(X_test[i]) < 50]

    # select 10 examples
    n = 10
    np.random.seed(0)
    idx = np.random.choice(short_examples, size=n, replace=False)

    # update test data
    X_test = [X_test[i] for i in idx]

    # fit and initialize predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn('class', clf, preproc=preprocessor)

    # setup perturbation options
    perturb_opts = {
        "filling": "parallel",
        "sample_proba": 0.5,
        "temperature": 1.0,
        "top_n": 100,
        "frac_mask_templates": 0.1,
        "stopwords": stopwords,
        "punctuation": punctuation,
    }

    # initialize exaplainer
    explainer = AnchorText(predictor=predictor, sampling_strategy=AnchorText.SAMPLING_LANGUAGE_MODEL,
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
@pytest.mark.parametrize('filling', ['parallel'])
@pytest.mark.parametrize('sample_proba', [0.5, 0.6, 0.7])
def test_lm_stopwords_punctuation(lang_model, punctuation, stopwords, filling,
                                  movie_sentiment_data, sample_proba, nlp):
    """
    Checks if the sampling procedure affect the stopwords and the
    punctuation when is not supposed to.
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
        raw, data = sampler.create_mask((), num_samples=10, filling=filling, **perturb_opts)

        for j in range(len(raw)):
            mask_counts = str(raw[j]).count(lang_model.mask)
            raw[j] = str(raw[j]).replace(lang_model.mask, '', mask_counts)

            preprocessed = nlp(str(raw[j]))
            words_masks = [w.text.lower() for w in preprocessed]
            words_masks = {w: words_masks.count(w) for w in words_masks}

            # check if the stopwords were perturbed and they were not supposed to
            for sw in stopwords:
                if sw in words:
                    assert words[sw] == words_masks[sw]

            # check if the punctuation was perturbed and it was not supposed to
            for p in punctuation:
                if (p in text) and (p != '\''):  # ' is a tricky one as in words don't for which don' is a token
                    assert text.count(p) == str(raw[j]).count(p)


@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
@pytest.mark.parametrize('head_gt, num_tokens',
                         [('word1', 2),
                          ('word1 word2', 3),
                          ('word1 word2 word3', 4),
                          ('word1 word2 word3 word4', 5),
                          ('word1 word2 word3 word4 word5', 6)])
def test_lm_split(lang_model, head_gt, num_tokens):
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
@pytest.mark.parametrize('filling', ['parallel', 'autoregressive'])
def test_lm_mask(lang_model, num_tokens, sample_proba, filling):
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
        "filling": filling,
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
@pytest.mark.parametrize('filling', ['parallel'])
@pytest.mark.parametrize('punctuation', [string.punctuation])
@pytest.mark.parametrize('sample_proba', [0.5, 0.6, 0.7])
def test_lm_sample_punctuation(lang_model, punctuation, filling, movie_sentiment_data, sample_proba):
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
        "filling": filling,
        "sample_punctuation": False
    }

    # initialize sampler
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

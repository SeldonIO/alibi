import numpy as np

import pytest
from pytest_lazyfixture import lazy_fixture

from alibi.explainers import AnchorText
from alibi.utils.download import spacy_model
from alibi.explainers.tests.utils import predict_fcn
from alibi.explainers.anchor_text import Neighbors, _load_spacy_lexeme_prob
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR


@pytest.mark.parametrize('text, n_punctuation_marks, n_unique_words',
                            [
                                ('This is a good book.', 1, 6),
                                ('I, for one, hate it.', 3, 7)
                            ]
                         )
@pytest.mark.parametrize('lr_classifier', [lazy_fixture('movie_sentiment_data')], indirect=True)
@pytest.mark.parametrize('predict_type, anchor, sampling_method, filling_method, threshold',
                            [
                                ('proba', (), 'unknown', None, 0.95),
                                ('proba', (), 'similarity', None, 0.95),
                                ('proba', (), 'language_model', 'parallel', 0.95)
                            ]
                         )
@pytest.mark.parametrize('lang_model',
                         ["None", "RobertaBase", "BertBaseUncased", "DistilbertBaseUncased"],
                         indirect=True
                         )
def test_explainer(text, n_punctuation_marks, n_unique_words, lr_classifier, predict_type,
                    anchor, sampling_method, filling_method, threshold, lang_model, nlp):
    cond1 = (sampling_method != 'language_model') and (lang_model is not None)
    cond2 = (sampling_method == 'language_model') and (lang_model is None)
    if any([cond1, cond2]):
        pytest.skip("Invalid combination")

    # test parameters
    num_samples = 100
    sample_proba = 0.5
    top_n = 500
    temperature = 1
    n_covered_ex = 5  # number of examples where the anchor applies to be retruned

    # fit and initialize predictor
    clf, preprocessor = lr_classifier
    predictor = predict_fcn(predict_type, clf, preproc=preprocessor)

    # test expalainer initalization
    explainer = AnchorText(nlp=nlp, language_model=lang_model, predictor=predictor)
    assert explainer.predictor(['book']).shape == (1, )

    # setup explainer
    perturb_opts = {
        'sampling_method': sampling_method,
        'filling_method': filling_method,
        'sample_proba': sample_proba,
        'temperature': temperature,
        'top_n': top_n,
    }

    # set sampler
    explainer.n_covered_ex = n_covered_ex
    explainer.set_sampler_perturbation(text=text, perturb_opts=perturb_opts)

    # set instance label
    if predict_type == 'proba':
        label = np.argmax(predictor([text])[0])
    else:
        label = predictor([text])[0]
    explainer.instance_label = label

    # check punctuation and words for `unknown` and `similarity` sampling
    if sampling_method in [AnchorText.SAMPLING_UNKNOWN, AnchorText.SAMPLING_SIMILARITY]:
        assert len(explainer.perturbation.punctuation) == n_punctuation_marks
        assert len(explainer.perturbation.words) == len(explainer.perturbation.positions)

    # test sampler
    cov_true, cov_false, labels, data, coverage, _ = explainer.sampler((0, anchor), num_samples)
    if not anchor:
        assert coverage == -1

    # check that words in present are in the proposed anchor
    if (sampling_method in [AnchorText.SAMPLING_SIMILARITY]) and len(anchor) > 0:
        assert len(anchor) * data.shape[0] == data[:, anchor].sum()

    if sampling_method == AnchorText.SAMPLING_UNKNOWN:
        all_words = explainer.perturbation.words
        assert len(np.unique(all_words)) == n_unique_words

    # test explanation
    explanation = explainer.explain(
        text,
        threshold=threshold,
        **perturb_opts
    )

    assert explanation.precision >= threshold
    assert explanation.raw['prediction'].item() == label
    assert explanation.meta.keys() == DEFAULT_META_ANCHOR.keys()
    assert explanation.data.keys() == DEFAULT_DATA_ANCHOR.keys()


import pytest
from pytest_lazyfixture import lazy_fixture
from alibi.utils.lang_model import *


@pytest.mark.parametrize('text, min_num',
                            [
                                ("This is ... a sentence, with a long ?!?, lot of punctuation; test this.", 5)
                            ]
                         )
@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
def test_punctuation(text, min_num, lang_model):
    """
    Checks if the language model can identify punctuation correctly
    """
    head, tail, head_tokens, tail_tokens = lang_model.head_tail_split(text)

    # assert head
    assert len(head) > 0
    assert len(head_tokens) > 0

    # assert tail
    assert tail is None
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
                            [(
                                'Test the following stopwords: this, but, a, the, verylongword',
                                ['this', 'but', 'a', 'the', 'verylongword']
                            )]
                         )
@pytest.mark.parametrize('lang_model', ['DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'], indirect=True)
def test_stopwords(text, stopwords, lang_model):
    """
    Checks if the language model can identify stopwords correctly
    """
    head, tail, head_tokens, tail_tokens = lang_model.head_tail_split(text)

    # head assertions
    assert len(head) > 0
    assert len(head_tokens) > 0

    # tail assertions
    assert tail is None
    assert len(tail_tokens) == 0

    # convert stopwords to lowercase
    stopwords = [s.strip().lower() for s in stopwords]

    # find all the subwords in the sentence
    found_stopwords = []

    for i in range(len(head_tokens)):
        if lang_model.is_stop_word(head_tokens, i, string.punctuation, stopwords):
            word = lang_model.select_entire_word(head_tokens, i, string.punctuation)
            found_stopwords.append(word.strip())

    # transform found_stopwords to lowercase
    found_stopwords = set([s.strip().lower() for s in found_stopwords])

    # check if the found subwords and the expected one are the same
    assert set(found_stopwords) == set(stopwords)

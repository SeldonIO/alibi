import logging
from abc import abstractmethod

from typing import (TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union)

import numpy as np
import spacy

if TYPE_CHECKING:
    import spacy  # noqa: F811

logger = logging.getLogger(__name__)


class Neighbors:
    def __init__(self, nlp_obj: 'spacy.language.Language', n_similar: int = 500, w_prob: float = -15.) -> None:
        """
        Initialize class identifying neighbouring words from the embedding for a given word.

        Parameters
        ----------
        nlp_obj
            `spaCy` model.
        n_similar
            Number of similar words to return.
        w_prob
            Smoothed log probability estimate of token's type.
        """
        self.nlp = nlp_obj
        self.w_prob = w_prob
        # list with spaCy lexemes in vocabulary
        # first if statement is a workaround due to some missing keys in models:
        # https://github.com/SeldonIO/alibi/issues/275#issuecomment-665017691
        self.to_check = [self.nlp.vocab[w] for w in self.nlp.vocab.vectors
                         if int(w) in self.nlp.vocab.strings and  # type: ignore[operator]
                         self.nlp.vocab[w].prob >= self.w_prob]
        self.n_similar = n_similar

    def neighbors(self, word: str, tag: str, top_n: int) -> dict:
        """
        Find similar words for a certain word in the vocabulary.

        Parameters
        ----------
        word
            Word for which we need to find similar words.
        tag
            Part of speech tag for the words.
        top_n
            Return only `top_n` neighbors.

        Returns
        -------
        A dict with two fields. The ``'words'`` field contains a `numpy` array of the `top_n` most similar words, \
        whereas the fields ``'similarities'`` is a `numpy` array with corresponding word similarities.
        """

        # the word itself is excluded so we add one to return the expected number of words
        top_n += 1

        texts: List = []
        similarities: List = []
        if word in self.nlp.vocab:
            word_vocab = self.nlp.vocab[word]
            queries = [w for w in self.to_check if w.is_lower == word_vocab.is_lower]
            if word_vocab.prob < self.w_prob:
                queries += [word_vocab]
            by_similarity = sorted(queries, key=lambda w: word_vocab.similarity(w), reverse=True)[:self.n_similar]

            # Find similar words with the same part of speech
            for lexeme in by_similarity:
                # because we don't add the word itself anymore
                if len(texts) == top_n - 1:
                    break
                token = self.nlp(lexeme.orth_)[0]
                if token.tag_ != tag or token.text == word:
                    continue
                texts.append(token.text)
                similarities.append(word_vocab.similarity(lexeme))

        words = np.array(texts) if texts else np.array(texts, dtype='<U')
        return {'words': words, 'similarities': np.array(similarities)}


def load_spacy_lexeme_prob(nlp: 'spacy.language.Language') -> 'spacy.language.Language':
    """
    This utility function loads the `lexeme_prob` table for a spacy model if it is not present.
    This is required to enable support for different spacy versions.
    """
    import spacy
    SPACY_VERSION = spacy.__version__.split('.')
    MAJOR, MINOR = int(SPACY_VERSION[0]), int(SPACY_VERSION[1])

    if MAJOR == 2:
        if MINOR < 3:
            return nlp
        elif MINOR == 3:
            # spacy 2.3.0 moved lexeme_prob into a different package `spacy_lookups_data`
            # https://github.com/explosion/spaCy/issues/5638
            try:
                table = nlp.vocab.lookups_extra.get_table('lexeme_prob')  # type: ignore[attr-defined]
                # remove the default empty table
                if table == dict():
                    nlp.vocab.lookups_extra.remove_table('lexeme_prob')  # type: ignore[attr-defined]
            except KeyError:
                pass
            finally:
                # access the `prob` of any word to load the full table
                assert nlp.vocab["a"].prob != -20.0, f"Failed to load the `lexeme_prob` table for model {nlp}"
    elif MAJOR >= 3:
        # in spacy 3.x we need to manually add the tables
        # https://github.com/explosion/spaCy/discussions/6388#discussioncomment-331096
        if 'lexeme_prob' not in nlp.vocab.lookups.tables:
            from spacy.lookups import load_lookups
            lookups = load_lookups(nlp.lang, ['lexeme_prob'])  # type: ignore[arg-type]
            nlp.vocab.lookups.add_table('lexeme_prob', lookups.get_table('lexeme_prob'))

    return nlp


class AnchorTextSampler:
    @abstractmethod
    def set_text(self, text: str) -> None:
        pass

    @abstractmethod
    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _joiner(self, arr: np.ndarray, dtype: Optional[Type[np.generic]] = None) -> np.ndarray:
        """
        Function to concatenate a `numpy` array of strings along a specified axis.

        Parameters
        ----------
        arr
            1D `numpy` array of strings.
        dtype
           Array type, used to avoid truncation of strings when concatenating along axis.

        Returns
        -------
        Array with one element, the concatenation of the strings in the input array.
        """
        if not dtype:
            return np.array(' '.join(arr))

        return np.array(' '.join(arr)).astype(dtype)


class UnknownSampler(AnchorTextSampler):
    UNK: str = "UNK"  #: Unknown token to be used.

    def __init__(self, nlp: 'spacy.language.Language', perturb_opts: Dict):
        """
        Initialize unknown sampler. This sampler replaces word with the `UNK` token.

        Parameters
        ----------
        nlp
            `spaCy` object.
        perturb_opts
            Perturbation options.
        """
        super().__init__()

        # set nlp and perturbation options
        self.nlp = load_spacy_lexeme_prob(nlp)
        self.perturb_opts: Union[Dict, None] = perturb_opts

        # define buffer for word, punctuation and position
        self.words: List = []
        self.punctuation: List = []
        self.positions: List = []

    def set_text(self, text: str) -> None:
        """
        Sets the text to be processed.

        Parameters
        ----------
        text
            Text to be processed.
        """
        # process text
        processed = self.nlp(text)  # spaCy tokens for text
        self.words = [x.text for x in processed]  # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]  # list with punctuation in text

        # set dtype
        self.set_data_type()

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns a `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by ``'UNK'`` token.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        raw
            Array containing num_samples elements. Each element is a perturbed sentence.
        data
            A `(num_samples, m)`-dimensional boolean array, where `m` is the number of tokens
            in the instance to be explained.
        """
        assert self.perturb_opts, "Perturbation options are not set."

        # allocate memory for the binary mask and the perturbed instances
        data = np.ones((num_samples, len(self.words)))
        raw = np.zeros((num_samples, len(self.words)), self.dtype)

        # fill each row of the raw data matrix with the text instance to be explained
        raw[:] = self.words

        for i, t in enumerate(self.words):
            # do not perturb words that are in anchor
            if i in anchor:
                continue

            # sample the words in the text outside of the anchor that are replaced with UNKs
            n_changed = np.random.binomial(num_samples, self.perturb_opts['sample_proba'])
            changed = np.random.choice(num_samples, n_changed, replace=False)
            raw[changed, i] = UnknownSampler.UNK
            data[changed, i] = 0

        # join the words
        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data

    def set_data_type(self) -> None:
        """
        Working with `numpy` arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """
        max_len = max(len(self.UNK), len(max(self.words, key=len)))
        max_sent_len = len(self.words) * max_len + len(self.UNK) * len(self.punctuation) + 1
        self.dtype = '<U' + str(max_sent_len)


class SimilaritySampler(AnchorTextSampler):

    def __init__(self, nlp: 'spacy.language.Language', perturb_opts: Dict):
        """
        Initialize similarity sampler. This sampler replaces words with similar words.

        Parameters
        ----------
        nlp
            `spaCy` object.
        perturb_opts
            Perturbation options.

        """
        super().__init__()

        # set nlp and perturbation options
        self.nlp = load_spacy_lexeme_prob(nlp)
        self.perturb_opts = perturb_opts

        # define synonym generator
        self._synonyms_generator = Neighbors(self.nlp)

        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.synonyms: Dict[str, Dict[str, np.ndarray]] = {}
        self.tokens: 'spacy.tokens.Doc'
        self.words: List[str] = []
        self.positions: List[int] = []
        self.punctuation: List['spacy.tokens.Token'] = []

    def set_text(self, text: str) -> None:
        """
        Sets the text to be processed

        Parameters
        ----------
        text
            Text to be processed.
        """
        processed = self.nlp(text)  # spaCy tokens for text
        self.words = [x.text for x in processed]  # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]  # punctuation in text
        self.tokens = processed

        # find similar words
        self.find_similar_words()

        # set dtype
        self.set_data_type()

    def find_similar_words(self) -> None:
        """
        This function queries a `spaCy` nlp model to find `n` similar words with the same
        part of speech for each word in the instance to be explained. For each word
        the search procedure returns a dictionary containing a `numpy` array of words (``'words'``)
        and a `numpy` array of word similarities (``'similarities'``).
        """
        for word, token in zip(self.words, self.tokens):
            if word not in self.synonyms:
                self.synonyms[word] = self._synonyms_generator.neighbors(word, token.tag_, self.perturb_opts['top_n'])

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns a `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by similar words with the same part of speech of tag.
        See :py:meth:`alibi.explainers.anchors.text_samplers.SimilaritySampler.perturb_sentence_similarity` for details
        of how the replacement works.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchors.text_samplers.SimilaritySampler.perturb_sentence_similarity`.
        """
        assert self.perturb_opts, "Perturbation options are not set."
        return self.perturb_sentence_similarity(anchor, num_samples, **self.perturb_opts)

    def perturb_sentence_similarity(self,
                                    present: tuple,
                                    n: int,
                                    sample_proba: float = 0.5,
                                    forbidden: frozenset = frozenset(),
                                    forbidden_tags: frozenset = frozenset(['PRP$']),
                                    forbidden_words: frozenset = frozenset(['be']),
                                    temperature: float = 1.,
                                    pos: frozenset = frozenset(['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']),
                                    use_proba: bool = False,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb the text instance to be explained.

        Parameters
        ----------
        present
            Word index in the text for the words in the proposed anchor.
        n
            Number of samples used when sampling from the corpus.
        sample_proba
            Sample probability for a word if `use_proba=False`.
        forbidden
            Forbidden lemmas.
        forbidden_tags
            Forbidden POS tags.
        forbidden_words
            Forbidden words.
        pos
            POS that can be changed during perturbation.
        use_proba
            Bool whether to sample according to a similarity score with the corpus embeddings.
        temperature
            Sample weight hyper-parameter if ``use_proba=True``.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        raw
            Array of perturbed text instances.
        data
            Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample.
        """
        # allocate memory for the binary mask and the perturbed instances
        raw = np.zeros((n, len(self.tokens)), self.dtype)
        data = np.ones((n, len(self.tokens)))

        # fill each row of the raw data matrix with the text to be explained
        raw[:] = [x.text for x in self.tokens]

        for i, t in enumerate(self.tokens):  # apply sampling to each token
            # if the word is part of the anchor, move on to next token
            if i in present:
                continue

            # check that token does not fall in any forbidden category
            if (t.text not in forbidden_words and t.pos_ in pos and
                    t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):

                t_neighbors = self.synonyms[t.text]['words']
                # no neighbours with the same tag or word not in spaCy vocabulary
                if t_neighbors.size == 0:
                    continue

                n_changed = np.random.binomial(n, sample_proba)
                changed = np.random.choice(n, n_changed, replace=False)

                if use_proba:  # use similarity scores to sample changed tokens
                    weights = self.synonyms[t.text]['similarities']
                    weights = np.exp(weights / temperature)  # weighting by temperature (check previous implementation)
                    weights = weights / sum(weights)
                else:
                    weights = np.ones((t_neighbors.shape[0],))
                    weights /= t_neighbors.shape[0]

                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights, replace=True)
                data[changed, i] = 0

        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data

    def set_data_type(self) -> None:
        """
        Working with `numpy` arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """
        max_len = 0
        max_sent_len = 0

        for word in self.words:
            similar_words = self.synonyms[word]['words']
            max_len = max(max_len, int(similar_words.dtype.itemsize /
                                       np.dtype(similar_words.dtype.char + '1').itemsize))
            max_sent_len += max_len
            self.dtype = '<U' + str(max_sent_len)

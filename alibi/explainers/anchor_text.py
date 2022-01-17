import copy
import logging
import string
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Type, Union)

import numpy as np
import spacy
import tensorflow as tf

from alibi.api.defaults import DEFAULT_DATA_ANCHOR, DEFAULT_META_ANCHOR
from alibi.api.interfaces import Explainer, Explanation
from alibi.exceptions import (AlibiPredictorCallException,
                              AlibiPredictorReturnTypeError)
from alibi.utils.lang_model import LanguageModel
from alibi.utils.wrappers import ArgmaxTransformer

from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation

if TYPE_CHECKING:
    import spacy  # noqa: F811
logger = logging.getLogger(__name__)


def _load_spacy_lexeme_prob(nlp: 'spacy.language.Language') -> 'spacy.language.Language':
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

        texts, similarities = [], []  # type: List, List
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
        self.nlp = _load_spacy_lexeme_prob(nlp)
        self.perturb_opts = perturb_opts  # type: Union[Dict, None]

        # define buffer for word, punctuation and position
        self.words, self.punctuation, self.positions = [], [], []  # type: List, List, List

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
        self.nlp = _load_spacy_lexeme_prob(nlp)
        self.perturb_opts = perturb_opts

        # define synonym generator
        self._synonyms_generator = Neighbors(self.nlp)

        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.synonyms = {}  # type: Dict[str, Dict[str, np.ndarray]]
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
        See :py:meth:`alibi.explainers.anchor_text.SimilaritySampler.perturb_sentence_similarity` for details of how
        the replacement works.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchor_text.SimilaritySampler.perturb_sentence`.
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


class LanguageModelSampler(AnchorTextSampler):
    # filling procedures
    FILLING_PARALLEL: str = 'parallel'  #: Parallel filling procedure.
    FILLING_AUTOREGRESSIVE = 'autoregressive'  #: Autoregressive filling procedure. Considerably slow.

    def __init__(self, model: LanguageModel, perturb_opts: dict, ):
        """
        Initialize language model sampler. This sampler replaces words with the ones
        sampled according to the output distribution of the language model. There are
        two modes to use the sampler: ``'parallel'`` and ``'autoregressive'``. In the ``'parallel'``
        mode, all words are replaced simultaneously. In the ``'autoregressive'`` model, the words
        are replaced one by one, starting from left to right. Thus the following words
        are conditioned on the previous predicted words.

        Parameters
        ----------
        model
            Transformers masked language model.
        perturb_opts
            Perturbation options.
        """
        super().__init__()

        # set language model and perturbation options
        self.model = model
        self.perturb_opts = perturb_opts

        # Define language model's vocab
        vocab: Dict[str, int] = self.model.tokenizer.get_vocab()

        # Define masking sampling tensor. This tensor is used to avoid sampling
        # certain tokens from the vocabulary such as: subwords, punctuation, etc.
        self.subwords_mask = np.zeros(len(vocab.keys()), dtype=np.bool_)

        for token in vocab:
            # Add subwords in the sampling mask. This means that subwords
            # will not be considered when sampling for the masked words.
            if self.model.is_subword_prefix(token):
                self.subwords_mask[vocab[token]] = True

            # Add punctuation in the sampling mask. This means that the
            # punctuation will not be considered when sampling for the masked words.
            sample_punctuation: bool = perturb_opts.get('sample_punctuation', False)
            punctuation: str = perturb_opts.get('punctuation', string.punctuation)

            if (not sample_punctuation) and self.model.is_punctuation(token, punctuation):
                self.subwords_mask[vocab[token]] = True

        # define head, tail part of the text
        self.head, self.tail = '', ''  # type: str, str
        self.head_tokens, self.tail_tokens = [], []  # type: List[str], List[str]

    def get_sample_ids(self,
                       punctuation: str = string.punctuation,
                       stopwords: Optional[List[str]] = None,
                       **kwargs) -> None:
        """
        Find indices in words which can be perturbed.

        Parameters
        ----------
        punctuation
            String of punctuation characters.
        stopwords
            List of stopwords.
        **kwargs
            Other arguments. Not used.
        """
        # transform stopwords to lowercase
        if stopwords:
            stopwords = [w.lower().strip() for w in stopwords]

        # Initialize list of indices allowed to be perturbed
        ids_sample = list(np.arange(len(self.head_tokens)))

        # Define partial function for stopwords checking
        is_stop_word = partial(
            self.model.is_stop_word,
            tokenized_text=self.head_tokens,
            punctuation=punctuation,
            stopwords=stopwords
        )

        # lambda expressions to check for a subword
        subword_cond = lambda token, idx: self.model.is_subword_prefix(token)  # noqa: E731
        # lambda experssion to check for a stopword
        stopwords_cond = lambda token, idx: is_stop_word(start_idx=idx)  # noqa: E731
        # lambda expression to check for punctuation
        punctuation_cond = lambda token, idx: self.model.is_punctuation(token, punctuation)  # noqa: E731

        # Gather all in a list of conditions
        conds = [punctuation_cond, stopwords_cond, subword_cond]

        # Remove indices of the tokens that are not allowed to be masked
        for i, token in enumerate(self.head_tokens):
            if any([cond(token, i) for cond in conds]):
                ids_sample.remove(i)

        # Save the indices allowed to be masked and the corresponding mapping.
        # The anchor base algorithm alters indices one by one. By saving the mapping
        # and sending only the initial token of a word, we avoid unnecessary sampling.
        # E.g. word = token1 ##token2. Instead of trying two anchors (1 0), (1, 1) - which are
        # equivalent because we take the full word, just try one (1)
        self.ids_sample = np.array(ids_sample)
        self.ids_mapping = {i: id for i, id in enumerate(self.ids_sample)}

    def set_text(self, text: str) -> None:
        """
        Sets the text to be processed

        Parameters
        ----------
        text
          Text to be processed.
        """
        # Some language models can only work with a limited number of tokens. Thus the text needs
        # to be split in head_text and tail_text. We will only alter the head_tokens.
        self.head, self.tail, self.head_tokens, self.tail_tokens = self.model.head_tail_split(text)

        # define indices of the words which can be perturbed
        self.get_sample_ids(**self.perturb_opts)

        # Set dtypes
        self.set_data_type()

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns a `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by words sampled according to the language
        model's predictions.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See :py:meth:`alibi.explainers.anchor_text.LanguageModelSampler.perturb_sentence`.
        """
        assert self.perturb_opts, "Perturbation options are not set."
        return self.perturb_sentence(anchor, num_samples, **self.perturb_opts)

    def perturb_sentence(self,
                         anchor: tuple,
                         num_samples: int,
                         sample_proba: float = .5,
                         top_n: int = 100,
                         batch_size_lm: int = 32,
                         filling: str = "parallel",
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns an `numpy` array of `num_samples` where randomly chosen features,
        except those in anchor, are replaced by words sampled according to the language
        model's predictions.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.
        sample_proba:
            Probability of a token being replaced by a similar token.
        top_n:
            Used for top n sampling.
        batch_size_lm:
            Batch size used for language model.
        filling:
            Method to fill masked words. Either ``'parallel'`` or ``'autoregressive'``.
        **kwargs
            Other arguments to be passed to other methods.

        Returns
        -------
        raw
            Array containing `num_samples` elements. Each element is a perturbed sentence.
        data
            A `(num_samples, m)`-dimensional boolean array, where `m` is the number of tokens
            in the instance to be explained.
        """
        # Create the mask
        raw, data = self.create_mask(
            anchor=anchor,
            num_samples=num_samples,
            sample_proba=sample_proba,
            filling=filling,
            **kwargs
        )

        # If the anchor does not cover the entire sentence,
        # then fill in mask with language model
        if len(anchor) != len(self.ids_sample):
            raw, data = self.fill_mask(
                raw=raw, data=data,
                num_samples=num_samples,
                top_n=top_n,
                batch_size_lm=batch_size_lm,
                filling=filling,
                **kwargs
            )

        # append tail if it exits
        raw = self._append_tail(raw) if self.tail else raw
        return raw, data

    def create_mask(self,
                    anchor: tuple,
                    num_samples: int,
                    sample_proba: float = 1.0,
                    filling: str = 'parallel',
                    frac_mask_templates: float = 0.1,
                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create mask for words to be perturbed.

        Parameters
        ----------
        anchor
            Indices represent the positions of the words to be kept unchanged.
        num_samples
            Number of perturbed sentences to be returned.
        sample_proba
            Probability of a word being replaced.
        filling:
            Method to fill masked words. Either ``'parallel'`` or ``'autoregressive'``.
        frac_mask_templates
            Fraction of mask templates from the number of requested samples.
        **kwargs
            Other arguments to be passed to other methods.

        Returns
        -------
        raw
            Array with masked instances.
        data
            A `(num_samples, m)`-dimensional boolean array, where `m` is the number of tokens
            in the instance to be explained.
        """
        # make sure that frac_mask_templates is in [0, 1]
        frac_mask_templates = np.clip(frac_mask_templates, 0, 1).item()

        # compute indices allowed be masked
        all_indices = range(len(self.ids_sample))
        allowed_indices = list(set(all_indices) - set(anchor))

        if len(allowed_indices) == 0 or filling == self.FILLING_AUTOREGRESSIVE:
            # If the anchor covers all the words that can be perturbed (it can happen)
            # then the number of mask_templates should be equal to the number of sampled requested.
            # If the filling is autoregressive, just generate from the start a `num_sample`
            # masks, cause the computation performance is pretty similar.
            mask_templates = num_samples
        else:
            # If the probability of sampling a word is 1, then all words will be masked.
            # Thus there is no point in generating more than one mask.
            # Otherwise compute the number of masking templates according to the fraction
            # passed as argument and make sure that at least one mask template is generated
            mask_templates = 1 if np.isclose(sample_proba, 1) else max(1, int(num_samples * frac_mask_templates))

        # allocate memory
        data = np.ones((mask_templates, len(self.ids_sample)))
        raw = np.zeros((mask_templates, len(self.head_tokens)), dtype=self.dtype_token)

        # fill each row of the raw data matrix with the text instance to be explained
        raw[:] = self.head_tokens

        # create mask
        if len(allowed_indices):
            for i in range(mask_templates):
                # Here the sampling of the indices of the word to be masked is done by rows
                # and not by columns as in the other sampling methods. The reason is that
                # is much easier to ensure that at least one word in the sentence is masked.
                # If the sampling is performed over the columns it might be the case
                # that no word in a sentence will be masked.
                n_changed = max(1, np.random.binomial(len(allowed_indices), sample_proba))
                changed = np.random.choice(allowed_indices, n_changed, replace=False)

                # mark the entrance as maks
                data[i, changed] = 0

                # Mask the corresponding words. This requires a mapping from indices
                # to the actual position of the words in the text
                changed_mapping = [self.ids_mapping[j] for j in changed]
                raw[i, changed_mapping] = self.model.mask

                # Have to remove the subwords of the masked word, which has to be done iteratively
                for j in changed_mapping:
                    self._remove_subwords(raw=raw, row=i, col=j, **kwargs)

        # join words
        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype_sent)
        return raw, data

    def _append_tail(self, raw: np.ndarray) -> np.ndarray:
        """
        Appends the tail part of the text to the new sampled head.

        Parameters
        ----------
        raw
            New sampled heads.

        Returns
        -------
        full_raw
            Concatenation of the new sampled head with the original tail.
        """
        full_raw = []

        for i in range(raw.shape[0]):
            new_head_tokens = self.model.tokenizer.tokenize(raw[i])
            new_tokens = new_head_tokens + self.tail_tokens
            full_raw.append(self.model.tokenizer.convert_tokens_to_string(new_tokens))

        # convert to array and return
        return np.array(full_raw, dtype=self.dtype_sent)

    def _joiner(self, arr: np.ndarray, dtype: Optional[Type[np.generic]] = None) -> np.ndarray:
        """
        Function to concatenate an `numpy` array of strings along a specified axis.

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
        filtered_arr = list(filter(lambda x: len(x) > 0, arr))
        str_arr = self.model.tokenizer.convert_tokens_to_string(filtered_arr)

        if not dtype:
            return np.array(str_arr)

        return np.array(str_arr).astype(dtype)

    def fill_mask(self,
                  raw: np.ndarray,
                  data: np.ndarray,
                  num_samples: int,
                  top_n: int = 100,
                  batch_size_lm: int = 32,
                  filling: str = "parallel",
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill in the masked tokens with language model.

        Parameters
        ----------
        raw
            Array of mask templates.
        data
            Binary mask having 0 where the word was masked.
        num_samples
            Number of samples to be drawn.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        filling
            Method to fill masked words. Either ``'parallel'`` or ``'autoregressive'``.
        **kwargs
            Other paremeters to be passed to other methods.

        Returns
        -------
        raw
            Array containing `num_samples` elements. Each element is a perturbed sentence.
        """
        # chose the perturbation function
        perturb_func = self._perturb_instances_parallel if filling == self.FILLING_PARALLEL \
            else self._perturb_instance_ar

        # perturb instances
        tokens, data = perturb_func(raw=raw, data=data,
                                    num_samples=num_samples,
                                    batch_size_lm=batch_size_lm,
                                    top_n=top_n, **kwargs)

        # decode the tokens and remove special characters as <pad>, <cls> etc.
        raw = self.model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return np.array(raw), data

    def _remove_subwords(self, raw: np.ndarray, row: int, col: int, punctuation: str = '', **kwargs) -> np.ndarray:
        """
        Deletes the subwords that follow a given token identified by the `(row, col)` pair in the `raw` matrix.
        A token is considered to be part of a word if is not a punctuation and if has the subword prefix
        specific to the used language model. The subwords are not actually deleted in, but they are replace
        by the empty string ``''``.

        Parameters
        ----------
        raw
            Array of tokens.
        row
            Row coordinate of the word to be removed.
        col
            Column coordinate of the word to be removed.
        punctuation
            String containing the punctuation to be considered.

        Returns
        -------
        raw
            Array of tokens where deleted subwords are replaced by the empty string.
        """
        for next_col in range(col + 1, len(self.head_tokens)):
            # if encounter a punctuation, just stop
            if self.model.is_punctuation(raw[row, next_col], punctuation):
                break

            # if it is a subword prefix, then replace it by empty string
            if self.model.is_subword_prefix(raw[row, next_col]):
                raw[row, next_col] = ''
            else:
                break

        return raw

    def _perturb_instances_parallel(self,
                                    num_samples: int,
                                    raw: np.ndarray,
                                    data: np.ndarray,
                                    top_n: int = 100,
                                    batch_size_lm: int = 32,
                                    temperature: float = 1.0,
                                    use_proba: bool = False,
                                    **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb the instances in a single forward pass (parallel).

        Parameters
        ----------
        num_samples
            Number of samples to be generated
        raw
            Array of mask templates. Has `mask_templates` rows.
        data
            Binary array having 0 where the tokens are masked. Has `mask_templates` rows.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        temperature
            Sample weight hyper-parameter.
        use_proba
            Bool whether to sample according to the predicted words distribution
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        sampled_tokens
            Array containing the ids of the sampled tokens. Has `num_samples` rows.
        sampled_data
            Binary array having 0 where the tokens were masked. Has `num_samples` rows.
        """
        # tokenize instances
        tokens_plus = self.model.tokenizer.batch_encode_plus(list(raw), padding=True, return_tensors='tf')

        # number of samples to generate per mask template
        remainder = num_samples % len(raw)
        mult_factor = num_samples // len(raw)

        # fill in masks with language model
        # (mask_template x max_length_sentence x num_tokens)
        logits = self.model.predict_batch_lm(x=tokens_plus,
                                             vocab_size=self.model.tokenizer.vocab_size,
                                             batch_size=batch_size_lm)

        # select rows and cols where the input the tokens are masked
        tokens = tokens_plus['input_ids']  # (mask_template x max_length_sentence)
        mask_pos = tf.where(tokens == self.model.mask_id)
        mask_row, mask_col = mask_pos[:, 0], mask_pos[:, 1]

        # buffer containing sampled tokens
        sampled_tokens = np.zeros((num_samples, tokens.shape[1]), dtype=np.int32)
        sampled_data = np.zeros((num_samples, data.shape[1]))

        for i in range(logits.shape[0]):
            # select indices corresponding to the current row `i`
            idx = tf.reshape(tf.where(mask_row == i), shape=-1)

            # select columns corresponding to the current row `i`
            cols = tf.gather(mask_col, idx)

            # select the logits of the masked input
            logits_mask = logits[i, cols, :]

            # mask out tokens according to the subword_mask
            logits_mask[:, self.subwords_mask] = -np.inf

            # select top n tokens from each distribution
            top_k = tf.math.top_k(logits_mask, top_n)
            top_k_logits, top_k_tokens = top_k.values, top_k.indices
            top_k_logits = (top_k_logits / temperature) if use_proba else (top_k_logits * 0)

            # sample `num_samples` instance for the current mask template
            for j in range(mult_factor + int(i < remainder)):
                # Compute the buffer index
                idx = i * mult_factor + j + min(i, remainder)

                # Sample indices
                ids_k = tf.reshape(tf.random.categorical(top_k_logits, 1), shape=-1)

                # Set the unmasked tokens and for the masked one and replace them with the samples drawn
                sampled_tokens[idx] = tokens[i]
                sampled_tokens[idx, cols] = tf.gather(top_k_tokens, ids_k, batch_dims=1)

            # Add the original binary mask which marks the beginning of a masked
            # word, as is needed for the anchor algorithm (backend stuff)
            idx, offset = i * mult_factor, min(i, remainder)
            sampled_data[idx + offset:idx + mult_factor + offset + (i < remainder)] = data[i]

        # Check that there are not masked tokens left
        assert np.all(sampled_tokens != self.model.mask_id)
        assert np.all(np.any(sampled_tokens != 0, axis=1))
        return sampled_tokens, sampled_data

    def _perturb_instance_ar(self,
                             num_samples: int,
                             raw: np.ndarray,
                             data: np.ndarray,
                             top_n: int = 100,
                             batch_size: int = 32,
                             temperature: float = 1.0,
                             use_proba: bool = False,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb the instances in an autoregressive fashion (sequential).

        Parameters
        ----------
        num_samples
            Number of samples to be generated.
        raw
            Array of mask templates. Has `mask_templates` rows.
        data
            Binary array having 0 where the tokens are masked. Has `mask_templates` rows.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        temperature
            Sample weight hyper-parameter.
        use_proba
            Bool whether to sample according to the predicted words distribution.
        **kwargs
            Other arguments. Not used.

        Returns
        -------
        sampled_tokens
            Array containing the ids of the sampled tokens. Has `num_samples` rows.
        sampled_data
            Binary array having 0 where the tokens were masked. Has `num_samples` rows.
        """
        # number of samples to generate per mask template
        assert num_samples == raw.shape[0]

        # tokenize instances
        tokens_plus = self.model.tokenizer.batch_encode_plus(list(raw), padding=True, return_tensors='tf')
        tokens = tokens_plus['input_ids'].numpy()  # (mask_template x max_length_sentence)

        # store the column indices for each row where a token is a mask
        masked_idx = []
        max_len_idx = -1
        mask_pos = tf.where(tokens == self.model.mask_id)
        mask_row, mask_col = mask_pos[:, 0], mask_pos[:, 1]

        for i in range(tokens.shape[0]):
            # get the columns indexes and store them in the buffer
            idx = tf.reshape(tf.where(mask_row == i), shape=-1)
            cols = tf.gather(mask_col, idx)
            masked_idx.append(cols)

            # update maximum length
            max_len_idx = max(max_len_idx, len(cols))

        # iterate through all possible columns indexes
        for i in range(max_len_idx):
            masked_rows, masked_cols = [], []

            # iterate through all possible examples
            for row in range(tokens.shape[0]):
                # this means that the row does not have any more masked columns
                if len(masked_idx[row]) <= i:
                    continue

                masked_rows.append(row)
                masked_cols.append(masked_idx[row][i])

            # compute logits
            logits = self.model.predict_batch_lm(x=tokens_plus,
                                                 vocab_size=self.model.tokenizer.vocab_size,
                                                 batch_size=batch_size)

            # select only the logits of the first masked word in each row
            logits_mask = logits[masked_rows, masked_cols, :]

            # mask out words according to the subword_mask
            logits_mask[:, self.subwords_mask] = -np.inf

            # select top n tokens from each distribution
            top_k = tf.math.top_k(logits_mask, top_n)
            top_k_logits, top_k_tokens = top_k.values, top_k.indices
            top_k_logits = (top_k_logits / temperature) if use_proba else (top_k_logits * 0)

            # Sample indices
            ids_k = tf.reshape(tf.random.categorical(top_k_logits, 1), shape=-1)

            # replace masked tokens with the sampled one
            tokens[masked_rows, masked_cols] = tf.gather(top_k_tokens, ids_k, batch_dims=1)
        return tokens, data

    def set_data_type(self) -> None:
        """
        Working with `numpy` arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """

        # get the vocabulary
        vocab = self.model.tokenizer.get_vocab()
        max_len = 0

        # go through the vocabulary and compute the maximum length of a token
        for token in vocab.keys():
            max_len = len(token) if len(token) > max_len else max_len

        # length of the maximum word. the prefix it is just a precaution.
        # for example <mask> -> _<mask> which is not in the vocabulary.
        max_len += len(self.model.SUBWORD_PREFIX)

        # length of the maximum text
        max_sent_len = (len(self.head_tokens) + len(self.tail_tokens)) * max_len

        # define the types to be used
        self.dtype_token = '<U' + str(max_len)
        self.dtype_sent = '<U' + str(max_sent_len)


DEFAULT_SAMPLING_UNKNOWN = {
    "sample_proba": 0.5
}
"""
Default perturbation options for ``'unknown'`` sampling

    - ``'sample_proba'`` : ``float`` - probability of a word to be masked.
"""

DEFAULT_SAMPLING_SIMILARITY = {
    "sample_proba": 0.5,
    "top_n": 100,
    "temperature": 1.0,
    "use_proba": False
}
"""
Default perturbation options for ``'similarity'`` sampling

    - ``'sample_proba'`` : ``float`` - probability of a word to be masked.

    - ``'top_n'`` : ``int`` - number of similar words to sample for perturbations.

    - ``'temperature'`` : ``float`` - sample weight hyper-parameter if `use_proba=True`.

    - ``'use_proba'`` : ``bool`` - whether to sample according to the words similarity.
"""

DEFAULT_SAMPLING_LANGUAGE_MODEL = {
    "filling": "parallel",
    "sample_proba": 0.5,
    "top_n": 100,
    "temperature": 1.0,
    "use_proba": False,
    "frac_mask_templates": 0.1,
    "batch_size_lm": 32,
    "punctuation": string.punctuation,
    "stopwords": [],
    "sample_punctuation": False,
}
"""
Default perturbation options for ``'language_model'`` sampling

    - ``'filling'`` : ``str`` - filling method for language models. Allowed values: ``'parallel'``, \
    ``'autoregressive'``. ``'parallel'`` method corresponds to a single forward pass through the language model. The \
    masked words are sampled independently, according to the selected probability distribution (see `top_n`, \
    `temperature`, `use_proba`). `autoregressive` method fills the words one at the time. This corresponds to \
    multiple forward passes through  the language model which is computationally expensive.

    - ``'sample_proba'`` : ``float`` - probability of a word to be masked.

    - ``'top_n'`` : ``int`` - number of similar words to sample for perturbations.

    - ``'temperature'`` : ``float`` - sample weight hyper-parameter if use_proba equals ``True``.

    - ``'use_proba'`` : ``bool`` - whether to sample according to the predicted words distribution. If set to \
    ``False``, the `top_n` words are sampled uniformly at random.

    - ``'frac_mask_template'`` : ``float`` - fraction from the number of samples of mask templates to be generated. \
    In each sampling call, will generate `int(frac_mask_templates * num_samples)` masking templates. \
    Lower fraction corresponds to lower computation time since the batch fed to the language model is smaller. \
    After the words' distributions is predicted for each mask, a total of `num_samples` will be generated by sampling \
    evenly from each template. Note that lower fraction might correspond to less diverse sample. A `sample_proba=1` \
    corresponds to masking each word. For this case only one masking template will be constructed. \
    A `filling='autoregressive'` will generate `num_samples` masking templates regardless of the value \
    of `frac_mask_templates`.

    - ``'batch_size_lm'`` : ``int`` - batch size used for the language model forward pass.

    - ``'punctuation'`` : ``str`` - string of punctuation not to be masked.

    - ``'stopwords'`` : ``List[str]`` - list of words not to be masked.

    - ``'sample_punctuation'`` : ``bool`` - whether to sample punctuation to fill the masked words. If ``False``, the \
    punctuation defined in `punctuation` will not be sampled.
"""


class AnchorText(Explainer):
    # sampling methods
    SAMPLING_UNKNOWN = 'unknown'  #: Unknown sampling strategy.
    SAMPLING_SIMILARITY = 'similarity'  #: Similarity sampling strategy.
    SAMPLING_LANGUAGE_MODEL = 'language_model'  #: Language model sampling strategy.

    # default params
    DEFAULTS: Dict[str, Dict] = {
        SAMPLING_UNKNOWN: DEFAULT_SAMPLING_UNKNOWN,
        SAMPLING_SIMILARITY: DEFAULT_SAMPLING_SIMILARITY,
        SAMPLING_LANGUAGE_MODEL: DEFAULT_SAMPLING_LANGUAGE_MODEL,
    }

    # class of samplers
    CLASS_SAMPLER = {
        SAMPLING_UNKNOWN: UnknownSampler,
        SAMPLING_SIMILARITY: SimilaritySampler,
        SAMPLING_LANGUAGE_MODEL: LanguageModelSampler
    }

    def __init__(self,
                 predictor: Callable[[List[str]], np.ndarray],
                 sampling_strategy: str = 'unknown',
                 nlp: Optional['spacy.language.Language'] = None,
                 language_model: Optional[LanguageModel] = None,
                 seed: int = 0,
                 **kwargs: Any) -> None:
        """
        Initialize anchor text explainer.

        Parameters
        ----------
        predictor
            A callable that takes a list of text strings representing `N` data points as inputs and returns `N` outputs.
        sampling_strategy
            Perturbation distribution method:

             - ``'unknown'`` - replaces words with UNKs.

             - ``'similarity'`` - samples according to a similarity score with the corpus embeddings.

             - ``'language_model'`` - samples according the language model's output distributions.

        nlp
            `spaCy` object when sampling method is ``'unknown'`` or ``'similarity'``.
        language_model
            Transformers masked language model. This is a model that it adheres to the
            `LanguageModel` interface we define in :py:class:`alibi.utils.lang_model.LanguageModel`.
        seed
            If set, ensure identical random streams.
        kwargs
            Sampling arguments can be passed as `kwargs` depending on the `sampling_strategy`.
            Check default arguments defined in:

                - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_UNKNOWN`

                - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_SIMILARITY`

                - :py:data:`alibi.explainers.anchor_text.DEFAULT_SAMPLING_LANGUAGE_MODEL`

        Raises
        ------
        :py:class:`alibi.exceptions.AlibiPredictorCallException`
            If calling `predictor` fails at runtime.
        :py:class:`alibi.exceptions.AlibiPredictorReturnTypeError`
            If the return type of `predictor` is not `np.ndarray`.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ANCHOR))
        self._seed(seed)

        # set the predictor
        self.predictor = self._transform_predictor(predictor)

        # define model which can be either spacy object or LanguageModel
        # the initialization of the model happens in _validate_kwargs
        self.model: Union['spacy.language.Language', LanguageModel]  #: Language model to be used.

        # validate kwargs
        self.perturb_opts, all_opts = self._validate_kwargs(sampling_strategy=sampling_strategy, nlp=nlp,
                                                            language_model=language_model, **kwargs)

        # set perturbation
        self.perturbation: Any = \
            self.CLASS_SAMPLER[self.sampling_strategy](self.model, self.perturb_opts)  #: Perturbation method.

        # update metadata
        self.meta['params'].update(seed=seed)
        self.meta['params'].update(**all_opts)

    def _validate_kwargs(self,
                         sampling_strategy: str,
                         nlp: Optional['spacy.language.Language'] = None,
                         language_model: Optional[LanguageModel] = None,
                         **kwargs: Any) -> Tuple[dict, dict]:

        # set sampling method
        sampling_strategy = sampling_strategy.strip().lower()
        sampling_strategies = [
            self.SAMPLING_UNKNOWN,
            self.SAMPLING_SIMILARITY,
            self.SAMPLING_LANGUAGE_MODEL
        ]

        # validate sampling method
        if sampling_strategy not in sampling_strategies:
            sampling_strategy = self.SAMPLING_UNKNOWN
            logger.warning(f"Sampling method {sampling_strategy} if not valid. "
                           f"Using the default value `{self.SAMPLING_UNKNOWN}`")

        if sampling_strategy in [self.SAMPLING_UNKNOWN, self.SAMPLING_SIMILARITY]:
            if nlp is None:
                raise ValueError("spaCy model can not be `None` when "
                                 f"`sampling_strategy` set to `{sampling_strategy}`.")
            # set nlp object
            self.model = _load_spacy_lexeme_prob(nlp)
        else:
            if language_model is None:
                raise ValueError("Language model can not be `None` when "
                                 f"`sampling_strategy` set to `{sampling_strategy}`")
            # set language model object
            self.model = language_model
            self.model_class = type(language_model).__name__

        # set sampling method
        self.sampling_strategy = sampling_strategy

        # get default args
        default_args: dict = self.DEFAULTS[self.sampling_strategy]
        perturb_opts: dict = deepcopy(default_args)  # contains only the perturbation params
        all_opts = deepcopy(default_args)  # contains params + some potential incorrect params

        # compute common keys
        allowed_keys = set(perturb_opts.keys())
        provided_keys = set(kwargs.keys())
        common_keys = allowed_keys & provided_keys

        # incorrect keys
        if len(common_keys) < len(provided_keys):
            incorrect_keys = ", ".join(provided_keys - common_keys)
            logger.warning("The following keys are incorrect: " + incorrect_keys)

        # update defaults args and all params
        perturb_opts.update({key: kwargs[key] for key in common_keys})
        all_opts.update(kwargs)
        return perturb_opts, all_opts

    def sampler(self, anchor: Tuple[int, tuple], num_samples: int, compute_labels: bool = True) -> \
            Union[List[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]], List[np.ndarray]]:
        """
        Generate perturbed samples while maintaining features in positions specified in
        anchor unchanged.

        Parameters
        ----------
        anchor
             - ``int`` - the position of the anchor in the input batch.

             - ``tuple`` - the anchor itself, a list of words to be kept unchanged.

        num_samples
            Number of generated perturbed samples.
        compute_labels
            If ``True``, an array of comparisons between predictions on perturbed samples and
            instance to be explained is returned.

        Returns
        -------
        If ``compute_labels=True``, a list containing the following is returned

         - `covered_true` - perturbed examples where the anchor applies and the model prediction \
         on perturbation is the same as the instance prediction.

         - `covered_false` - perturbed examples where the anchor applies and the model prediction \
         is NOT the same as the instance prediction.

         - `labels` - num_samples ints indicating whether the prediction on the perturbed sample \
         matches (1) the label of the instance to be explained or not (0).

         - `data` - Matrix with 1s and 0s indicating whether a word in the text has been perturbed for each sample.

         - `-1.0` - indicates exact coverage is not computed for this algorithm.

         - `anchor[0]` - position of anchor in the batch request.

        Otherwise, a list containing the data matrix only is returned.
        """

        raw_data, data = self.perturbation(anchor[1], num_samples)

        # create labels using model predictions as true labels
        if compute_labels:
            labels = self.compare_labels(raw_data)
            covered_true = raw_data[labels][:self.n_covered_ex]
            covered_false = raw_data[np.logical_not(labels)][:self.n_covered_ex]

            # coverage set to -1.0 as we can't compute 'true' coverage for this model
            return [covered_true, covered_false, labels.astype(int), data, -1.0, anchor[0]]
        else:
            return [data]

    def compare_labels(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute the agreement between a classifier prediction on an instance to be explained
        and the prediction on a set of samples which have a subset of features fixed to a
        given value (aka compute the precision of anchors).

        Parameters
        ----------
        samples
            Samples whose labels are to be compared with the instance label.

        Returns
        -------
        A `numpy` boolean array indicating whether the prediction was the same as the instance label.
        """
        return self.predictor(samples.tolist()) == self.instance_label

    def explain(self,  # type: ignore[override]
                text: str,
                threshold: float = 0.95,
                delta: float = 0.1,
                tau: float = 0.15,
                batch_size: int = 100,
                coverage_samples: int = 10000,
                beam_size: int = 1,
                stop_on_first: bool = True,
                max_anchor_size: Optional[int] = None,
                min_samples_start: int = 100,
                n_covered_ex: int = 10,
                binary_cache_size: int = 10000,
                cache_margin: int = 1000,
                verbose: bool = False,
                verbose_every: int = 1,
                **kwargs: Any) -> Explanation:
        """
        Explain instance and return anchor with metadata.

        Parameters
        ----------
        text
            Text instance to be explained.
        threshold
            Minimum precision threshold.
        delta
            Used to compute `beta`.
        tau
            Margin between lower confidence bound and minimum precision or upper bound.
        batch_size
            Batch size used for sampling.
        coverage_samples
            Number of samples used to estimate coverage from during anchor search.
        beam_size
            Number of options kept after each stage of anchor building.
        stop_on_first
            If ``True``, the beam search algorithm will return the first anchor that has satisfies the
            probability constraint.
        max_anchor_size
            Maximum number of features to include in an anchor.
        min_samples_start
            Number of samples used for anchor search initialisation.
        n_covered_ex
            How many examples where anchors apply to store for each anchor sampled during search
            (both examples where prediction on samples agrees/disagrees with predicted label are stored).
        binary_cache_size
            The anchor search pre-allocates `binary_cache_size` batches for storing the boolean arrays
            returned during sampling.
        cache_margin
            When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache
            of the same size is pre-allocated to continue buffering samples.
        verbose
            Display updates during the anchor search iterations.
        verbose_every
            Frequency of displayed iterations during anchor search process.
        **kwargs
            Other keyword arguments passed to the anchor beam search and the text sampling and perturbation functions.

        Returns
        -------
        `Explanation` object containing the anchor explaining the instance with additional metadata as attributes. \
        Contains the following data-related attributes

         - `anchor` : ``List[str]`` - a list of words in the proposed anchor.

         - `precision` : ``float`` - the fraction of times the sampled instances where the anchor holds yields \
         the same prediction as the original instance. The precision will always be  threshold for a valid anchor.

         - `coverage` : ``float`` - the fraction of sampled instances the anchor applies to.
        """
        # get params for storage in meta
        params = locals()
        remove = ['text', 'self']
        for key in remove:
            params.pop(key)

        params = deepcopy(params)  # Get a reference to itself if not deepcopy for LM sampler

        # store n_covered_ex positive/negative examples for each anchor
        self.n_covered_ex = n_covered_ex
        self.instance_label = self.predictor([text])[0]

        # set sampler
        self.perturbation.set_text(text)

        # get anchors and add metadata
        mab = AnchorBaseBeam(
            samplers=[self.sampler],
            sample_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            **kwargs
        )

        result = mab.anchor_beam(
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            desired_confidence=threshold,
            max_anchor_size=max_anchor_size,
            min_samples_start=min_samples_start,
            beam_size=beam_size,
            coverage_samples=coverage_samples,
            stop_on_first=stop_on_first,
            verbose=verbose,
            verbose_every=verbose_every,
            **kwargs,
        )  # type: Any

        if self.sampling_strategy == self.SAMPLING_LANGUAGE_MODEL:
            # take the whole word (this points just to the first part of the word)
            result['positions'] = [self.perturbation.ids_mapping[i] for i in result['feature']]
            result['names'] = [
                self.perturbation.model.select_word(
                    self.perturbation.head_tokens,
                    idx_feature,
                    self.perturbation.perturb_opts['punctuation']
                ) for idx_feature in result['positions']
            ]
        else:
            result['names'] = [self.perturbation.words[x] for x in result['feature']]
            result['positions'] = [self.perturbation.positions[x] for x in result['feature']]

        # set mab
        self.mab = mab
        return self._build_explanation(text, result, self.instance_label, params)

    def _build_explanation(self, text: str, result: dict, predicted_label: int, params: dict) -> Explanation:
        """
        Uses the metadata returned by the anchor search algorithm together with
        the instance to be explained to build an explanation object.

        Parameters
        ----------
        text
            Instance to be explained.
        result
            Dictionary containing the search result and metadata.
        predicted_label
            Label of the instance to be explained. Inferred if not received.
        params
            Arguments passed to `explain`.
        """

        result['instance'] = text
        result['instances'] = [text]  # TODO: should this be an array?
        result['prediction'] = np.array([predicted_label])
        exp = AnchorExplanation('text', result)

        # output explanation dictionary
        data = copy.deepcopy(DEFAULT_DATA_ANCHOR)
        data.update(anchor=exp.names(),
                    precision=exp.precision(),
                    coverage=exp.coverage(),
                    raw=exp.exp_map)

        # create explanation object
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=data)

        # params passed to explain
        # explanation.meta['params'].update(params)
        return explanation

    def _transform_predictor(self, predictor: Callable) -> Callable:
        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        x = ['Hello world']
        try:
            prediction = predictor(x)
        except Exception as e:
            msg = f"Predictor failed to be called on x={x}. " \
                  f"Check that `predictor` works with inputs of type List[str]."
            raise AlibiPredictorCallException(msg) from e

        if not isinstance(prediction, np.ndarray):
            msg = f"Excepted predictor return type to be {np.ndarray} but got {type(prediction)}."
            raise AlibiPredictorReturnTypeError(msg)

        if np.argmax(prediction.shape) == 0:
            return predictor
        else:
            transformer = ArgmaxTransformer(predictor)
            return transformer

    def reset_predictor(self, predictor: Callable) -> None:
        """
        Resets the predictor function.

        Parameters
        ----------
        predictor
            New predictor function.
        """
        self.predictor = self._transform_predictor(predictor)

    def _seed(self, seed: int) -> None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

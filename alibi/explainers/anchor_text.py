import sys
import copy
import torch
import string
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Union, Optional

from alibi.utils.wrappers import ArgmaxTransformer
from alibi.utils.lang_model import LanguageModel

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR
from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation

if TYPE_CHECKING:
    import spacy
logger = logging.getLogger(__name__)


def _load_spacy_lexeme_prob(nlp: 'spacy.language.Language'):
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
                table = nlp.vocab.lookups_extra.get_table('lexeme_prob')
                # remove the default empty table
                if table == dict():
                    nlp.vocab.lookups_extra.remove_table('lexeme_prob')
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
            lookups = load_lookups(nlp.lang, ['lexeme_prob'])
            nlp.vocab.lookups.add_table('lexeme_prob', lookups.get_table('lexeme_prob'))  # type: ignore

    return nlp


class Neighbors(object):

    def __init__(self, nlp_obj: 'spacy.language.Language', n_similar: int = 500, w_prob: float = -15.) -> None:
        """
        Initialize class identifying neighbouring words from the embedding for a given word.

        Parameters
        ----------
        nlp_obj
            spaCy model
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
        self.to_check = [self.nlp.vocab[w] for w in self.nlp.vocab.vectors if
                         int(w) in self.nlp.vocab.strings and self.nlp.vocab[w].prob >= self.w_prob]
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
            Return only top_n neighbors.

        Returns
        -------
        A dict with two fields. The 'words' field contains a numpy array
        of the top_n most similar words, whereas the fields similarity is
        a numpy array with corresponding word similarities.
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
    def set_params(self, text: str, perturb_opts: dict):
        pass

    @abstractmethod
    def __call__(self, anchor: tuple, num_samples: int):
        pass

    def _joiner(self, arr: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
        """
        Function to concatenate an np.array of strings along a specified axis.

        Parameters
        ----------
        arr
            1D numpy array of strings.
        dtype
           Array type, used to avoid truncation of strings when concatenating along axis.

        Returns
        -------
            Array with one element, the concatenation of the strings in the input array.
        """
        if not dtype:
            return np.array(' '.join(arr))

        return np.array(' '.join(arr)).astype(dtype)


class UnkownSampler(AnchorTextSampler):
    UNK = "UNK"

    def __init__(self, nlp: 'spacy.language.Language'):
        """
        Initialize unknown sampler. This sampler replaces word with the `UNK` token.

        Parameters
        ----------
        nlp
            spaCy model.
        """
        super(UnkownSampler, self).__init__()

        # store nlp object
        self.nlp = nlp

        # define buffer for word, punctuation and position
        self.words, self.punctuation, self.positions = [], [], [] # type: List, List, List

        # define perturbation options
        self.perturb_opts = None  # type: Union[Dict, None]

    def set_params(self, text: str, perturb_opts: dict):
        """
        Sets the text to be processed and the perturbation options

        Parameters
        ----------
        text
            Text to be processed.
        perturb_opts:
            Perturbation options.
        """
        # process text
        processed = self.nlp(text)                                  # spaCy tokens for text
        self.words = [x.text for x in processed]                    # list with words in text
        self.positions = [x.idx for x in processed]                 # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]     # list with punctuation in text

        # set perturbation options
        self.perturb_opts = perturb_opts

        # set dtype
        self.set_data_type()

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns  an np.array of num_samples where randomly chose features
        except those in anchor are replaced by self.UNK token.

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
            A (num_samples, m)-dimensional boolean array, where m is the number of tokens
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
            raw[changed, i] = UnkownSampler.UNK
            data[changed, i] = 0

        # join the words
        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)
        return raw, data

    def set_data_type(self) -> None:
        """
        Working with numpy arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.
        """
        max_len = max(len(self.UNK), len(max(self.words, key=len)))
        max_sent_len = len(self.words) * max_len + len(self.UNK) * len(self.punctuation) + 1
        self.dtype = '<U' + str(max_sent_len)


class SimilaritySampler(AnchorTextSampler):
    def __init__(self, nlp: 'spacy.language.Language'):
        """
        Initialize similarity sampler. This sampler replaces words with similar words.
        """
        super(SimilaritySampler, self).__init__()

        # store nlp object
        self.nlp = nlp

        # define synomym generator
        self._synonyms_generator = Neighbors(self.nlp)

        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.synonyms = {}  # type: Dict[str, Dict[str, np.ndarray]]
        self.tokens, self.words, self.positions, self.punctuation = [], [], [], []  # type: List, List, List, List

        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.synonyms = {}  # type: Dict[str, Dict[str, np.ndarray]]

        # the method used to generate samples
        self.perturbation = None  # type: Union[Callable, None]

        # perturbation_options
        self.perturb_opts = None  # type: Union[Dict, None]

    def set_params(self, text: str, perturb_opts: dict):
        """
        Sets the text to be processed and the perturbation options

        Parameters
        ----------
        text
          Text to be processed.
        perturb_opts
          Perturbation options.
        """
        processed = self.nlp(text)                                  # spaCy tokens for text
        self.words = [x.text for x in processed]                    # list with words in text
        self.positions = [x.idx for x in processed]                 # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]     # punctuation in text
        self.tokens = processed

        # set perturbation options
        self.perturb_opts = perturb_opts

        # find similar words
        self.find_similar_words()

        # set dtype
        self.set_data_type()

    def find_similar_words(self) -> None:
        """
        This function queries a spaCy nlp model to find n similar words with the same
        part of speech for each word in the instance to be explained. For each word
        the search procedure returns a dictionary containing an np.array of words ('words')
        and an np.array of word similarities ('similarities').
        """
        for word, token in zip(self.words, self.tokens):
            if word not in self.synonyms:
                self.synonyms[word] = self._synonyms_generator.neighbors(word, token.tag_, self.perturb_opts['top_n'])

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns  an np.array of num_samples where randomly chose features
        except those in anchor are replaced by similar words with the same part of
        speech of tag. See self.perturb_sentence for details of how the replacement works.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See perturb_sentence_similarity
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
                                    use_similarity_proba: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perturb the text instance to be explained.

        Parameters
        ----------
        present
            Word index in the text for the words in the proposed anchor.
        n
            Number of samples used when sampling from the corpus.
        sample_proba
            Sample probability for a word if use_similarity_proba is False.
        forbidden
            Forbidden lemmas.
        forbidden_tags
            Forbidden POS tags.
        forbidden_words
            Forbidden words.
        pos
            POS that can be changed during perturbation.
        use_similarity_proba
            Bool whether to sample according to a similarity score with the corpus embeddings.
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True.

        Returns
        -------
        raw
            Array of perturbed text instances.
        data
            Matrix with 1s and 0s indicating whether a word in the text
            has not been perturbed for each sample.
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

                if use_similarity_proba:  # use similarity scores to sample changed tokens
                    weights = self.synonyms[t.text]['similarities']
                    weights = weights ** (1. / temperature)  # weighting by temperature
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
        Working with numpy arrays of strings requires setting the data type to avoid
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
    # filling methods
    FILLING_PARALLEL = 'parallel'
    FILLING_AUTOREGRESSIVE = 'autoregressive'

    def __init__(self, model: LanguageModel):
        """
        Initialize language model sampler. This sampler replaces words with the ones
        sampled according to the output distribution of the language model. There are
        two modes to use the sampler: `parallel` and `autoregressive`. In the parallel
        mode, all words are replaced at ones. In the `autoregressive` model, the words
        are replaced one by one, starting from left to right. Thus the following words
        are conditioned also on the previous predicted words.

        Parameters
        ----------
        model
            Language model to be used to perturb the sentences.
        """
        super(LanguageModelSampler, self).__init__()

        # store language model
        self.model = model

        # define perturbation options
        self.perturb_opts = None  # type: Union[Dict, None]

        # define head, tail part of the text
        self.head, self.tail = '', ''  # type: str, str
        self.head_tokens, self.tail_tokens = None, None  # type: Union[List[str], None], Union[List[str], None]

    def get_sample_ids(self,
                       perturb_punctuation: bool = False,
                       perturb_stopwords: bool = False,
                       punctuation: str = string.punctuation,
                       stopwords: Optional[List[str]] = None,
                       **kwargs) -> None:
        """
        Find indices in words which can be perturbed.

        Parameters
        ----------
        perturb_punctuation
            Whether to allow punctuations to be perturbed.
        perturb_stopwords
            Whether to allow stopwords to be perturbed.
        punctuation
            String of punctuation characters.
        stopwords
            List of stopwords.
        """
        # Initialize list of indices allowed to be perturbed
        ids_sample = list(np.arange(len(self.head_tokens)))

        # Define partial function for stopwords checking
        is_stop_word = partial(
            self.model.is_stop_word,
            text=self.head_tokens,
            punctuation=punctuation,
            stopwords=stopwords
        )

        # lambda expressions to check for a subword
        subword_cond = lambda token, idx: self.model.is_subword_prefix(token)

        # lambda experssion to check for a stopword
        stopwords_cond = lambda token, idx: (not perturb_stopwords) and is_stop_word(start_idx=idx)

        # lambda expression to check for punctuation
        punctuation_cond = lambda token, idx: (not perturb_punctuation) and \
                self.model.is_punctuation(token, punctuation)

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

    def set_params(self, text: str, perturb_opts: dict):
        """
        Sets the text to be processed and the perturbation options

        Parameters
        ----------
        text
          Text to be processed.
        perturb_opts
          Perturbation options.
        """
        # Set perturbation options
        self.perturb_opts = perturb_opts

        # Some language models can only work with a limited number of tokens. Thus the text needs
        # to be split in head_text and tail_text. We will only alter the head_tokens.
        self.head, self.tail, self.head_tokens, self.tail_tokens = self.model.head_tail_split(text)

        # Set dtypes
        self.set_data_type()

        # Define language model's vocab
        vocab: Dict[str, int] = self.model.tokenizer.get_vocab()

        # Define masking sampling tensor. This tensor is used to avoid sampling
        # certain tokens from the vocabulary such as: subwords, punctuation, etc.
        self.subwords_mask = torch.zeros(len(vocab.keys()), dtype=torch.bool)

        for token in vocab:
            # Add subwords in the sampling mask. This means that subwords
            # will not be considered when sampling for the masked words.
            if self.model.is_subword_prefix(token):
                self.subwords_mask[vocab[token]] = True

            # Add punctuation in the sampling mask. This means that the
            # punctuation will not be considered when sampling for the masked words.
            if not perturb_opts.get('sample_punctuation', False) and \
                    self.model.is_punctuation(token, perturb_opts.get('punctuation', string.punctuation)):
                self.subwords_mask[vocab[token]] = True

        # define indices of the words which can be perturbed
        self.get_sample_ids(**perturb_opts)

    def __call__(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns  an np.array of num_samples where randomly chose features
        except those in anchor are replaced by words sampled form the language model
        prediction.

        Parameters
        ----------
        anchor:
            Indices represent the positions of the words to be kept unchanged.
        num_samples:
            Number of perturbed sentences to be returned.

        Returns
        -------
        See perturb_sentence
        """
        assert self.perturb_opts, "Perturbation options are not set."
        return self.perturb_sentence(anchor, num_samples, **self.perturb_opts)

    def perturb_sentence(self,
                         anchor: tuple,
                         num_samples: int,
                         sample_proba: float = .5,
                         top_n: int = 1,
                         batch_size_lm: int = 32,
                         filling_method: str = "parallel",
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        The function returns  an np.array of num_samples where randomly chose features
        except those in anchor are replaced by similar words.

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
        filling_method:
            Method to fill masked words. Either `parallel` or `ar`.

        Returns
        -------
        raw
            Array containing num_samples elements. Each element is a perturbed sentence.
        data
            A (num_samples, m)-dimensional boolean array, where m is the number of tokens
            in the instance to be explained.
        """
        # Create the mask
        data, raw = self.create_mask(
            anchor=anchor,
            num_samples=num_samples,
            sample_proba=sample_proba,
            filling_method=filling_method,
            **kwargs
        )

        # If the anchor does not covers the entire sentence,
        # then fill in mask with language model
        if len(anchor) != len(self.ids_sample):
            raw, data = self.fill_mask(
                raw=raw, data=data,
                num_samples=num_samples,
                top_n=top_n,
                batch_size_lm=batch_size_lm,
                filling_method=filling_method,
                **kwargs
            )
        
        # append tail if it exits
        raw = self._append_tail(raw) if self.tail else raw
        return raw, data

    def create_mask(self,
                    anchor: tuple,
                    num_samples: int,
                    sample_proba: float = 1.0,
                    filling_method: str = 'parallel',
                    prec_mask_templates: float = 1.0,
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
        filling_method:
            Filling method procedure. Can be `parallel` or `autoregressive`.
            See constructor for more details.
        prec_mask_templates
            Procentage form the number of samples of mask templates

        Returns
        -------
        data
            A (num_samples, m)-dimensional boolean array, where m is the number of tokens
            in the instance to be explained.
        raw
            List with masked instances.
        """
        # make sure that prec_mask_templates is in [0, 1]
        prec_mask_templates = np.clip(prec_mask_templates, 0, 1)

        # compute indices allowed be masked
        all_indices = range(len(self.ids_sample))
        allowed_indices = list(set(all_indices) - set(anchor))

        if len(allowed_indices) == 0 or filling_method == self.FILLING_AUTOREGRESSIVE:
            # If the anchor covers all the word that can be perturbed (it can happen)
            # then the number of mask_templates should be equal to the number of sampled requested.
            # If the filling method is autoregressive, just generate from the start a `num_sample`
            # masks, cause the computation performance is pretty similar.
            mask_templates = num_samples
        else:
            # If the probability of sampling a word is 1, then all words will be masked.
            # Thus there is no point in generating more than one mask.
            # Otherwise compute the number of masking templates according to the precentage
            # passed as argument and make sure that at least one mask template is generated
            mask_templates = 1 if np.isclose(sample_proba, 1) else max(1, int(num_samples * prec_mask_templates))

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
        return data, raw

    def _append_tail(self, raw: np.ndarray) -> np.array:
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
        full_raw = np.array(full_raw, dtype=self.dtype_sent)
        return full_raw

    def _joiner(self, arr: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
        """
        Function to concatenate an np.array of strings along a specified axis.

        Parameters
        ----------
        arr
            1D numpy array of strings.
        dtype
           Array type, used to avoid truncation of strings when concatenating along axis.

        Returns
        -------
            Array with one element, the concatenation of the strings in the input array.
        """
        arr = list(filter(lambda x: len(x) > 0, arr))
        arr = self.model.tokenizer.convert_tokens_to_string(arr)

        if not dtype:
            return np.array(arr)

        return np.array(arr).astype(dtype)

    def fill_mask(self,
                  raw: np.ndarray,
                  data: np.ndarray,
                  num_samples: int,
                  top_n: int = 1,
                  batch_size_lm: int = 32,
                  filling_method: str = "parallel",
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill in the masked tokens with language model.

        Parameters
        ----------
        raw
            Array of mask templates.
        data
            Binary mask having 0 where the word was masked. Note that
            the width of the data array is equal to the length of id_samples.
        num_samples
            Number of samples to be drawn.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        filling_method
            Filling method procedure. Can be `parallel` or `autoregressive`.
            See constructor for more details.

        Returns
        -------
        raw
            Array containing num_samples elements. Each element is a perturbed sentence.
        """
        # chose the perturbation function
        perturb_func = self._perturb_instances_parallel if filling_method == self.FILLING_PARALLEL\
            else self._perturb_instance_ar

        # perturb instances
        tokens, data = perturb_func(
            raw=raw, data=data,
            num_samples=num_samples,
            batch_size_lm=batch_size_lm,
            top_n=top_n,
            **kwargs
        )
        # decode the tokens and remove special characters as <pad>, <cls> etc.
        raw = self.model.tokenizer.batch_decode(tokens, **dict(skip_special_tokens=True))
        return np.array(raw), data

    def _remove_subwords(self, raw: np.array, row: int, col: int,
                         punctuation: str = string.punctuation, **kwargs) -> np.array:
        """
        Deletes the subwords that follow a given token identified by the (row, col) pair in the `raw` matrix.
        A token is considered to be part of a word if is not a punctuation and if has the subword prefix
        specific to the used language model. The subwords are not actually deleted in, but they are replace
        by the empty string ``

        Parameters
        ----------
        raw
            Array of tokens.
        row
            Row coordinate of the word to be removed.
        col
            Col coodrinate of the word to be removed.
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
                                    top_n: int = 1,
                                    batch_size_lm: int = 32,
                                    temperature: float = 1.0,
                                    **kwargs) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Perturb the instances in a single forward pass (parallel).

        Parameters
        ----------
        num_samples
            Number of samples to be generates
        raw
            Array of mask templates. Has `mask_templates` rows
        data
            Binary array having 0 where the tokens are masked. Has `mask_templates` rows.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        temperature
            Sample weight hyper-parameter.

        Returns
        -------
        sampled_tokens
            Tensor containing the ids of the sampled tokens.
            Has `num_samples` rows.
        sampled_data
            Binary array havin 0 where the tokens were masked.
            Has `num_samples` rows.
        """
        # tokenize instances
        tokens_plus = self.model.tokenizer.batch_encode_plus(list(raw), padding=True, return_tensors='pt')

        # number of samples to generate per mask template
        reminder = num_samples % len(raw)
        mult_factor = num_samples // len(raw)

        # fill in masks with language model
        # (mask_template x max_length_sentence x num_tokens)
        logits = self.model.predict_batch_lm(
            x=tokens_plus,
            device=self.model.device,
            vocab_size=self.model.tokenizer.vocab_size,
            batch_size=batch_size_lm
        )

        # select rows and cols where the input the tokens are masked
        tokens = tokens_plus['input_ids']  # (mask_template x max_length_sentence)
        mask_row, mask_col = torch.where(tokens == self.model.mask_token)

        # buffer containing sampled tokens
        sampled_tokens = torch.zeros(num_samples, tokens.shape[1], dtype=torch.long)
        sampled_data = np.zeros((num_samples, data.shape[1]))

        # initialize offset
        offset = 0

        for i in range(logits.shape[0]):
            # select indices corresponding to the current row `i`
            idx = torch.where(mask_row == i)[0]

            # select columns corresponding to the current row `i`
            cols = mask_col[idx]

            # select the logits of the masked input
            logits_mask = logits[i, cols, :]

            # mask out tokens according to the subword_mask
            logits_mask[:, self.subwords_mask] = -np.inf

            # select top n tokens from each distribution
            top_k = torch.topk(logits_mask, top_n, dim=1)
            top_k_logits, top_k_tokens = top_k.values, top_k.indices

            # create categorical distribution that we can sample the words from
            top_k_logits /= temperature
            dist = torch.distributions.Categorical(logits=top_k_logits)

            # sample `num_samples` instance for the current mask template
            for j in range(mult_factor + int(reminder > 0)):
                # Compute the buffer index
                idx = i * mult_factor + j + offset

                # Sample indices
                ids_k = dist.sample().reshape(-1, 1)

                # Set the unmasked tokens and for the masked one and replace them with the samples drawn
                sampled_tokens[idx] = tokens[i]
                sampled_tokens[idx, cols] = torch.reshape(top_k_tokens.gather(1, ids_k), (-1,))

            # Add the original binary mask which marks the beginning of a masked
            # word, as is needed for the anchor algorithm (backend stuff)
            idx = i * mult_factor
            sampled_data[idx + offset:idx + mult_factor + offset + (reminder > 0)] = data[i]

            # decrement reminder and increase counter if reminder is gt 0
            reminder -= 1
            offset += int(reminder > 0)

        # Check that there are not masked tokens left
        assert torch.all(sampled_tokens != self.model.mask_token)
        return sampled_tokens, sampled_data

    def _perturb_instance_ar(self,
                             num_samples: int,
                             raw: np.ndarray,
                             data: np.ndarray,
                             top_n: int = 1,
                             batch_size: int = 32,
                             temperature: float = 1.0,
                             **kwargs) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Perturb the instances in an autoregressive fashion (sequential).

        ----------
        num_samples
            Number of samples to be generates
        raw
            Array of mask templates. Has `mask_templates` rows
        data
            Binary array having 0 where the tokens are masked. Has `mask_templates` rows.
        top_n:
            Use the top n words when sampling.
        batch_size_lm:
            Batch size used for language model.
        temperature
            Sample weight hyper-parameter.

        Returns
        -------
        sampled_tokens
            Tensor containing the ids of the sampled tokens.
            Has `num_samples` rows.
        sampled_data
            Binary array havin 0 where the tokens were masked.
            Has `num_samples` rows.
        """

        # number of samples to generate per mask template
        assert num_samples == len(raw)

        # tokenize instances
        tokens_plus = self.model.tokenizer.batch_encode_plus(list(raw), padding=True, return_tensors='pt')
        tokens = tokens_plus['input_ids']  # (mask_template x max_length_sentence)

        # store the column indices for each row where a token is a mask
        masked_idx = []
        max_len_idx = -1
        mask_row, mask_col = torch.where(tokens == self.model.mask_token)

        for i in range(tokens.shape[0]):
            # get the columns indexes and store them in the buffer
            idx = torch.where(mask_row == i)[0]
            cols = mask_col[idx]
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
            logits = self.model.predict_batch_lm(
                tokens_plus,
                self.model.device,
                self.model.tokenizer.vocab_size,
                batch_size
            )

            # select only the logits of the first masked word in each row
            logits_mask = logits[masked_rows, masked_cols, :]

            # mask out words according to the subword_mask
            logits_mask[:, self.subwords_mask.long()] = -np.inf

            # select top n tokens from each distribution
            top_k = torch.topk(logits_mask, top_n, dim=1)
            top_k_logits, top_k_tokens = top_k.values, top_k.indices

            # create categorical distribution that we can sample the words from
            top_k_logits /= temperature
            dist = torch.distributions.Categorical(logits=top_k_logits)

            # sample indexes
            ids_k = dist.sample().reshape(-1, 1)

            # replace masked tokens with the sampled one
            tokens[masked_rows, masked_cols] = torch.reshape(top_k_tokens.gather(1, ids_k), (-1,))

        return tokens, data

    def set_data_type(self) -> None:
        """
        Working with numpy arrays of strings requires setting the data type to avoid
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
        # for example <mask> -> _<mask> which is not in the vocabulary).
        max_len += len(self.model.SUBWORD_PREFIX)

        # length of the maximum text
        max_sent_len = (len(self.head_tokens) + len(self.tail_tokens)) * max_len

        # define the types to be used
        self.dtype_token = '<U' + str(max_len)
        self.dtype_sent = '<U' + str(max_sent_len)


class AnchorText(Explainer):
    # sampling methods
    SAMPLING_UNKNOWN = 'unknown'
    SAMPLING_SIMILARITY = 'similarity'
    SAMPLING_LANGUAGE_MODEL = 'language_model'

    def __init__(self,
                 predictor: Callable,
                 nlp: Optional['spacy.language.Language'] = None,
                 language_model: Optional[LanguageModel] = None,
                 seed: int = None) -> None:
        """
        Initialize anchor text explainer.

        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        nlp
            spaCy object.
        seed
            If set, ensures identical random streams.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ANCHOR))
        np.random.seed(seed)

        # set nlp
        self.nlp = _load_spacy_lexeme_prob(nlp) if nlp else None

        # set the language model
        self.model = language_model

        # set the predictor
        self.predictor = self._transform_predictor(predictor)

        # the method used to generate samples
        self.perturbation = None  # type: Union[Callable, None]

        # update metadata
        self.meta['params'].update(seed=seed)

    def sampler(self, anchor: Tuple[int, tuple], num_samples: int, compute_labels: bool = True) -> \
            Union[List[Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]], List[np.ndarray]]:
        """
        Generate perturbed samples while maintaining features in positions specified in
        anchor unchanged.

        Parameters
        ----------
        anchor
            int: the position of the anchor in the input batch
            tuple: the anchor itself, a list of words to be kept unchanged
        num_samples
            Number of generated perturbed samples.
        compute_labels
            If True, an array of comparisons between predictions on perturbed samples and
            instance to be explained is returned.

        Returns
        -------
            If compute_labels=True, a list containing the following is returned:
             - covered_true: perturbed examples where the anchor applies and the model prediction
                    on perturbation is the same as the instance prediction
             - covered_false: perturbed examples where the anchor applies and the model prediction
                    is NOT the same as the instance prediction
             - labels: num_samples ints indicating whether the prediction on the perturbed sample
                    matches (1) the label of the instance to be explained or not (0)
             - data: Matrix with 1s and 0s indicating whether a word in the text has been
                     perturbed for each sample
             - 1.0: indicates exact coverage is not computed for this algorithm
             - anchor[0]: position of anchor in the batch request
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
            A boolean array indicating whether the prediction was the same as the instance label.
        """

        return self.predictor(samples.tolist()) == self.instance_label

    def set_sampler_perturbation(self, text: str, perturb_opts: dict) -> None:
        """
        Initialises the explainer by setting the perturbation function and
        parameters necessary to sample according to the perturbation method.

        Parameters
        ----------
        text
            Text to be explained
        perturb_opts
            A dict with keys:
                - 'sampling_method': TODO add more description            
                - 'sample_proba': given a feature and n sentences, this parameters is the mean of a Bernoulli \
                distribution used to decide how many sentences will have that feature perturbed
                - 'temperature': a tempature used to callibrate the softmax distribution over the sampling weights.
                
        """  # noqa W605

        sampling_methods = [self.SAMPLING_UNKNOWN, self.SAMPLING_SIMILARITY, self.SAMPLING_LANGUAGE_MODEL]
        if perturb_opts['sampling_method'] not in sampling_methods:
            perturb_opts['sampling_method'] = self.SAMPLING_UNKNOWN
            logger.warning(
                f'"sampling_method" unknown. Defaulting to "{self.SAMPLING_UNKNOWN}" behaviour.'
            )

        if perturb_opts['sampling_method'] == self.SAMPLING_SIMILARITY:
            self.perturbation = SimilaritySampler(self.nlp)
            self.perturbation.set_params(text, perturb_opts)

        elif perturb_opts['sampling_method'] == self.SAMPLING_LANGUAGE_MODEL:
            self.perturbation = LanguageModelSampler(self.model)
            self.perturbation.set_params(text, perturb_opts)

        else:
            self.perturbation = UnkownSampler(self.nlp)
            self.perturbation.set_params(text, perturb_opts)

    def explain(self,  # type: ignore
                text: str,
                sampling_method: str = 'unknown',
                filling_method: str = 'parallel',
                sample_proba: float = 0.5,
                top_n: int = 100,
                temperature: float = 1.,
                threshold: float = 0.95,
                delta: float = 0.1,
                tau: float = 0.15,
                batch_size: int = 100,
                coverage_samples: int = 10000,
                beam_size: int = 1,
                stop_on_first: bool = True,
                max_anchor_size: int = None,
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
        sampling_method
            Perturbation distribution method.
            `unknown` perturbation distribution will replace words randomly with UNKs.
            `similarity` sample according to a similarity scroe with the corpus embeddings.
            `language_model` sample according the distribution output by a language model.
        filling_method
            Filling method for language models. TODO add more descripiton
        masked_templates
            Number of tamplates to generae when using language models. TODO add more description
        sample_proba
            Sample probability if use_similarity_proba is False.
        top_n
            Number of similar words to sample for perturbations, only used if use_unk=False.
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True.
        threshold
            Minimum precision threshold.
        delta
            Used to compute beta.
        tau
            Margin between lower confidence bound and minimum precision or upper bound.
        batch_size
            Batch size used for sampling.
        coverage_samples
            Number of samples used to estimate coverage from during anchor search.
        beam_size
            Number of options kept after each stage of anchor building.
        stop_on_first
            If True, the beam search algorithm will return the first anchor that has satisfies the
            probability constraint.
        max_anchor_size
            Maximum number of features to include in an anchor.
        min_samples_start
            Number of samples used for anchor search initialisation.
        n_covered_ex
            How many examples where anchors apply to store for each anchor sampled during search
            (both examples where prediction on samples agrees/disagrees with predicted label are stored).
        binary_cache_size
            The anchor search pre-allocates binary_cache_size batches for storing the boolean arrays
            returned during sampling.
        cache_margin
            When only max(cache_margin, batch_size) positions in the binary cache remain empty, a new cache
            of the same size is pre-allocated to continue buffering samples.
        kwargs
            Other keyword arguments passed to the anchor beam search and the text sampling and perturbation functions.
        verbose
            Display updates during the anchor search iterations.
        verbose_every
            Frequency of displayed iterations during anchor search process.

        Returns
        -------
        explanation
            `Explanation` object containing the anchor explaining the instance with additional metadata as attributes.
        """
        # get params for storage in meta
        params = locals()
        remove = ['text', 'self']
        for key in remove:
            params.pop(key)

        # store n_covered_ex positive/negative examples for each anchor
        self.n_covered_ex = n_covered_ex
        self.instance_label = self.predictor([text])[0]

        # set the sampling function and type for samples' arrays
        perturb_opts = {
            'sampling_method': sampling_method,
            'filling_method': filling_method,
            'sample_proba': sample_proba,
            'temperature': temperature,
            'coverage_samples': coverage_samples,
            'top_n': top_n
        }
        perturb_opts.update(kwargs)

        # set sampler
        self.set_sampler_perturbation(text, perturb_opts)

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

        if sampling_method == self.SAMPLING_LANGUAGE_MODEL:
            # take the whole word (this takes just the first part of the word)
            features = [self.perturbation.ids_mapping[i] for i in result['feature']]
            result['names'] = [
                self.model.select_entire_word(
                    self.perturbation.head_tokens,
                    idx_feature,
                    perturb_opts.get('punctuation', string.punctuation)
                ) for idx_feature in features
            ]
            # TODO: see what happens to positions
        else:
            result['names'] = [self.perturbation.words[x] for x in result['feature']]
            result['positions'] = [self.perturbation.positions[x] for x in result['feature']]

        # set mab
        self.mab = mab

        # clear some attributes set during the `explain` call
        self._clear_state()

        return self.build_explanation(text, result, self.instance_label, params)

    def build_explanation(self, text: str, result: dict, predicted_label: int, params: dict) -> Explanation:
        """ Uses the metadata returned by the anchor search algorithm together with
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
            Parameters passed to `explain`
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
        explanation.meta['params'].update(params)
        return explanation

    def _transform_predictor(self, predictor: Callable) -> Callable:
        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        if np.argmax(predictor(['Hello world']).shape) == 0:
            return predictor
        else:
            transformer = ArgmaxTransformer(predictor)
            return transformer

    def reset_predictor(self, predictor: Callable) -> None:
        self.predictor = self._transform_predictor(predictor)

    def _clear_state(self):
        """
        Clears the explainer attributes set during the `explain` call. This is to avoid the explainer
        having state from a previous `explain` call which can also interfere with serializaiton.
        """
        # TODO: should we organize the state set during `explain` call into a single dictionary attribute
        #  instead of being scatter across several attributes?
        # TODO: currently not clearing self.meta which is updated during `explain`
        self.tokens, self.words, self.positions, self.punctuation = [], [], [], []
        self.synonyms = {}

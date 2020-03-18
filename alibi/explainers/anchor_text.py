import copy
import logging
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Union

from alibi.utils.wrappers import ArgmaxTransformer

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_ANCHOR, DEFAULT_DATA_ANCHOR
from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)


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
        self.to_check = [w for w in self.nlp.vocab if w.prob >= self.w_prob and w.has_vector]
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

        return {
            'words': np.array(texts),
            'similarities': np.array(similarities),
        }


class AnchorText(Explainer):
    UNK = 'UNK'

    def __init__(self, nlp: 'spacy.language.Language', predictor: Callable, seed: int = None) -> None:
        """
        Initialize anchor text explainer.

        Parameters
        ----------
        nlp
            spaCy object.
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        seed
            If set, ensures identical random streams.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_ANCHOR))
        np.random.seed(seed)

        self.nlp = nlp

        # check if predictor returns predicted class or prediction probabilities for each class
        # if needed adjust predictor so it returns the predicted class
        if np.argmax(predictor(['Hello world']).shape) == 0:
            self.predictor = predictor
        else:
            self.predictor = ArgmaxTransformer(predictor)

        self.neighbors = Neighbors(self.nlp)
        self.tokens, self.words, self.positions, self.punctuation = [], [], [], []  # type: List, List, List, List
        # dict containing an np.array of similar words with same part of speech and an np.array of similarities
        self.neighbours = {}  # type: Dict[str, Dict[str, np.ndarray]]
        # the method used to generate samples
        self.perturbation = None  # type: Callable

    def set_words_and_pos(self, text: str) -> None:
        """
        Process the sentence to be explained into spaCy token objects, a list of words,
        punctuation marks and a list of positions in input sentence.

        Parameters
        ----------
        text
            The instance to be explained.
        """

        processed = self.nlp(text)  # spaCy tokens for text
        self.words = [x.text for x in processed]  # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.punctuation = [x for x in processed if x.is_punct]
        self.tokens = processed

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
            # coverage set to -1.0 as we can't compute 'true'coverage for this model
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

    def set_sampler_perturbation(self, use_unk: bool, perturb_opts: dict) -> None:
        """
        Initialises the explainer by setting the perturbation function and
        parameters necessary to sample according to the perturbation method.

        Parameters
        ----------
        use_unk
            see explain method
        perturb_opts:
            A dict with keys:
                'top_n': the max number of alternatives to sample from for replacement
                'use_similarity_proba': if True the probability of selecting a replacement
                    word is prop. to the similarity between the word and the word to be replaced
                'sample_proba': given a feature and n sentences, this parameters is the mean of a
                    Bernoulli distribution used to decide how many sentences will have that feature
                    perturbed
                'temperature': a tempature used to callibrate the softmax distribution over the
                    sampling weights.
        """

        if use_unk and perturb_opts['use_similarity_proba']:
            logger.warning('"use_unk" and "use_similarity_proba" args should not both be True. '
                           'Defaulting to "use_unk" behaviour.')

        # Set object properties used by both samplers
        self.perturb_opts = perturb_opts
        self.sample_proba = perturb_opts['sample_proba']

        if use_unk:
            self.perturbation = self._unk
        else:
            self.find_similar_words()
            self.perturbation = self._similarity

    def _unk(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

        words = self.words
        data = np.ones((num_samples, len(words)))
        raw = np.zeros((num_samples, len(words)), self.dtype)
        # fill each row of the raw data matrix with the text instance to be explained
        raw[:] = words

        for i, t in enumerate(words):

            if i in anchor:
                continue

            # sample the words in the text outside of the anchor that are replaced with UNKs
            n_changed = np.random.binomial(num_samples, self.sample_proba)
            changed = np.random.choice(num_samples, n_changed, replace=False)
            raw[changed, i] = AnchorText.UNK
            data[changed, i] = 0

        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)

        return raw, data

    def _similarity(self, anchor: tuple, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
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
            see _unk method
        """
        return self.perturb_sentence(
            anchor,
            num_samples,
            **self.perturb_opts,
        )

    @staticmethod
    def _joiner(arr: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
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
        else:
            return np.array(' '.join(arr)).astype(dtype)

    def perturb_sentence(self, present: tuple, n: int, sample_proba: float = 0.5,
                         forbidden: frozenset = frozenset(), forbidden_tags: frozenset = frozenset(['PRP$']),
                         forbidden_words: frozenset = frozenset(['be']), temperature: float = 1.,
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
            Bool whether to sample according to a similarity score with the
            corpus embeddings.
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True.

        Returns
        -------
        raw_data
            Array of perturbed text instances.
        data
            Matrix with 1s and 0s indicating whether a word in the text
            has not been perturbed for each sample.
        """

        raw = np.zeros((n, len(self.tokens)), self.dtype)
        data = np.ones((n, len(self.tokens)))
        # fill each row of the raw data matrix with the text to be explained
        raw[:] = [x.text for x in self.tokens]

        for i, t in enumerate(self.tokens):  # apply sampling to each token

            if i in present:  # if the word is part of the anchor, move on to next token
                continue

            # check that token does not fall in any forbidden category
            if (t.text not in forbidden_words and t.pos_ in pos and
                    t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):

                t_neighbors = self.neighbours[t.text]['words']
                # no neighbours with the same tag or word not in spaCy vocabulary
                if t_neighbors.size == 0:
                    continue

                n_changed = np.random.binomial(n, sample_proba)
                changed = np.random.choice(n, n_changed, replace=False)

                if use_similarity_proba:  # use similarity scores to sample changed tokens
                    weights = self.neighbours[t.text]['similarities']
                    weights = weights ** (1. / temperature)  # weighting by temperature
                    weights = weights / sum(weights)
                else:
                    weights = np.ones((t_neighbors.shape[0],))
                    weights /= t_neighbors.shape[0]

                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights, replace=True)
                data[changed, i] = 0
        raw = np.apply_along_axis(self._joiner, axis=1, arr=raw, dtype=self.dtype)

        return raw, data

    def find_similar_words(self) -> None:
        """
        This function queries a spaCy nlp model to find n similar words with the same
        part of speech for each word in the instance to be explained. For each word
        the search procedure returns a dictionary containing an np.array of words ('words')
        and an np.array of word similarities ('similarities').
        """

        for word, token in zip(self.words, self.tokens):
            if word not in self.neighbours:
                self.neighbours[word] = self.neighbors.neighbors(word,
                                                                 token.tag_,
                                                                 self.perturb_opts['top_n'],
                                                                 )

    def set_data_type(self, use_unk: bool) -> None:
        """
        Working with numpy arrays of strings requires setting the data type to avoid
        truncating examples. This function estimates the longest sentence expected
        during the sampling process, which is used to set the number of characters
        for the samples and examples arrays. This depends on the perturbation method
        used for sampling.

        Parameters
        ----------
        use_unk
            See explain method.
        """

        # Compute the maximum sentence length as a function of the perturbation method
        max_sent_len, max_len = 0, 0
        if use_unk:
            max_len = max(len(self.UNK), len(max(self.words, key=len)))
            max_sent_len = len(self.words) * max_len + len(self.UNK) * len(self.punctuation) + 1
        else:
            for word in self.words:
                similar_words = self.neighbours[word]['words']
                max_len = max(max_len, int(similar_words.dtype.itemsize /
                                           np.dtype(similar_words.dtype.char + '1').itemsize))
                max_sent_len += max_len
        self.dtype = '<U' + str(max_sent_len)

    def explain(self,  # type: ignore
                text: str,
                use_unk: bool = True,
                use_similarity_proba: bool = False,
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
        use_unk
            If True, perturbation distribution will replace words randomly with UNKs.
            If False, words will be replaced by similar words using word embeddings.
        use_similarity_proba
            Sample according to a similarity score with the corpus embeddings
            use_unk needs to be False in order for this to be used.
        sample_proba
            Sample probability if use_similarity_proba is False.
        top_n
            Number of similar words to sample for perturbations, only used if use_proba=True.
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
            Dictionary containing the anchor explaining the instance with additional metadata.
        """
        # get params for storage in meta
        params = locals()
        remove = ['text', 'self']
        for key in remove:
            params.pop(key)

        # store n_covered_ex positive/negative examples for each anchor
        self.n_covered_ex = n_covered_ex
        self.instance_label = self.predictor([text])[0]
        # find words and their positions in the text instance
        self.set_words_and_pos(text)

        # set the sampling function and type for samples' arrays
        perturb_opts = {
            'use_similarity_proba': use_similarity_proba,
            'sample_proba': sample_proba,
            'temperature': temperature,
            'top_n': top_n,
        }
        perturb_opts.update(kwargs)
        self.set_sampler_perturbation(use_unk, perturb_opts)
        self.set_data_type(use_unk)

        # get anchors and add metadata
        mab = AnchorBaseBeam(
            samplers=[self.sampler],
            sample_cache_size=binary_cache_size,
            cache_margin=cache_margin,
            **kwargs)
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
        result['names'] = [self.words[x] for x in result['feature']]
        result['positions'] = [self.positions[x] for x in result['feature']]
        self.mab = mab

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
        result['prediction'] = predicted_label
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

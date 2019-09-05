from .anchor_base import AnchorBaseBeam
from .anchor_explanation import AnchorExplanation
import logging
import numpy as np
from typing import Any, Callable, Tuple, Dict, TYPE_CHECKING

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
            Number of similar words to return
        w_prob
            Smoothed log probability estimate of token's type
        """
        self.nlp = nlp_obj
        self.w_prob = w_prob
        self.to_check = [w for w in self.nlp.vocab if w.prob >= self.w_prob and w.has_vector]  # list with spaCy lexemes
        # in vocabulary
        self.n = {}  # type: Dict[str, list]
        self.n_similar = n_similar

    def neighbors(self, word: str) -> list:
        """
        Find similar words for a certain word in the vocabulary.

        Parameters
        ----------
        word
            Word for which we need to find similar words

        Returns
        -------
        A list containing tuples with the similar words and similarity scores.
        """
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []  # word not in vocabulary, so no info on neighbors
            else:
                word_vocab = self.nlp.vocab[word]
                queries = [w for w in self.to_check if w.is_lower == word_vocab.is_lower]
                if word_vocab.prob < self.w_prob:
                    queries += [word_vocab]
                # sort queries by similarity in descending order
                by_similarity = sorted(queries, key=lambda w: word_vocab.similarity(w), reverse=True)
                # store list of tuples containing the similar word and the word similarity ...
                # ... for nb of most similar words
                self.n[word] = [(self.nlp(w.orth_)[0], word_vocab.similarity(w))
                                for w in by_similarity[:self.n_similar]]
        return self.n[word]


class AnchorText(object):

    def __init__(self, nlp: 'spacy.language.Language', predict_fn: Callable) -> None:
        """
        Initialize anchor text explainer.

        Parameters
        ----------
        nlp
            spaCy object
        predict_fn
            Model prediction function
        """
        self.nlp = nlp

        # check if predict_fn returns predicted class or prediction probabilities for each class
        # if needed adjust predict_fn so it returns the predicted class
        if np.argmax(predict_fn(['Hello world']).shape) == 0:
            self.predict_fn = predict_fn
        else:
            self.predict_fn = lambda x: np.argmax(predict_fn(x), axis=1)

        self.neighbors = Neighbors(self.nlp)

    def get_sample_fn(self, text: str, desired_label: int = None, use_similarity_proba: bool = False,
                      use_unk: bool = True, sample_proba: float = 0.5, top_n: int = 100,
                      temperature: float = 0.4, **kwargs) -> Tuple[list, list, Callable]:
        """
        Create sampling function as well as lists with the words and word positions in the text.

        Parameters
        ----------
        text
            Text instance to be explained
        desired_label
            Label to use as true label for the instance to be explained
        use_similarity_proba
            Bool whether to sample according to a similarity score with the corpus embeddings
        use_unk
            If True, perturbation distribution will replace words randomly with UNKs.
            If False, words will be replaced by similar words using word embeddings.
        sample_proba
            Sample probability if use_similarity_proba is False
        top_n
            Sample using only top_n instances from similar words in the corpus
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True

        Returns
        -------
        words
            List with words in the text
        positions
            List with positions of the words in the text
        sample_fn
            Function returning perturbed text instances, matrix with flags for perturbed words and labels
        """
        # if no true label available; true label = predicted label
        true_label = desired_label
        if true_label is None:
            true_label = self.predict_fn([text])[0]

        processed = self.nlp(text)  # spaCy tokens for text
        words = [x.text for x in processed]  # list with words in text
        positions = [x.idx for x in processed]  # positions of words in text

        def sample_fn(present: list, num_samples: int, compute_labels: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                                                             np.ndarray]:
            """
            Create sampling function using similar words in the embedding space.

            Parameters
            ----------
            present
                List with the word index in the text for the words in the proposed anchor
            num_samples
                Number of samples used when sampling from the corpus
            compute_labels
                Boolean whether to use labels coming from model predictions as 'true' labels

            Returns
            -------
            raw_data
                num_samples of perturbed text instance
            data
                Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample
            labels
                Create labels using model predictions if compute_labels equals True
            """
            if use_unk:  # perturb examples by replacing words with UNKs

                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|S80')
                raw[:] = words  # fill each row of the raw data matrix with the text instance to be explained

                for i, t in enumerate(words):
                    if i in present:  # if the index corresponds to the index of a word in the anchor
                        continue

                    # sample the words in the text outside of the anchor that are replaced with UNKs
                    n_changed = np.random.binomial(num_samples, sample_proba)
                    changed = np.random.choice(num_samples, n_changed, replace=False)
                    raw[changed, i] = 'UNK'
                    data[changed, i] = 0

                # convert numpy array into list
                raw_data = [' '.join([y.decode() for y in x]) for x in raw]

            else:  # replace words by similar words instead of UNKs

                raw_data, data = self.perturb_sentence(text, present, num_samples, top_n=top_n,
                                                       use_similarity_proba=use_similarity_proba,
                                                       sample_proba=sample_proba, temperature=temperature,
                                                       **kwargs)

            # create labels using model predictions as true labels
            labels = np.array([])
            if compute_labels:
                labels = (self.predict_fn(raw_data) == true_label).astype(int)
            raw_data = np.array(raw_data).reshape(-1, 1)
            return raw_data, data, labels

        return words, positions, sample_fn

    def explain(self, text: str, threshold: float = 0.95, delta: float = 0.1,
                tau: float = 0.15, batch_size: int = 100, top_n: int = 100, desired_label: int = None,
                use_similarity_proba: bool = False, use_unk: bool = True,
                sample_proba: float = 0.5, temperature: float = 1., **kwargs: Any) -> dict:
        """
        Explain instance and return anchor with metadata.

        Parameters
        ----------
        text
            Text instance to be explained
        threshold
            Minimum precision threshold
        delta
            Used to compute beta
        tau
            Margin between lower confidence bound and minimum precision or upper bound
        batch_size
            Batch size used for sampling
        top_n
            Number of similar words to sample for perturbations, only used if use_proba=True
        desired_label
            Label to use as true label for the instance to be explained
        use_similarity_proba
            Bool whether to sample according to a similarity score with the corpus embeddings.
            use_unk needs to be False in order for use_similarity_proba equals True to be used.
        use_unk
            If True, perturbation distribution will replace words randomly with UNKs.
            If False, words will be replaced by similar words using word embeddings.
        sample_proba
            Sample probability if use_similarity_proba is False
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True
        kwargs
            Other keyword arguments passed to the anchor beam search and the text sampling and perturbation functions

        Returns
        -------
        explanation
            Dictionary containing the anchor explaining the instance with additional metadata
        """
        if use_unk and use_similarity_proba:
            logger.warning('"use_unk" and "use_similarity_proba" args should not both be True. '
                           'Defaults to "use_unk" behaviour.')

        # get the words and positions of words in the text instance and sample function
        words, positions, sample_fn = self.get_sample_fn(text, desired_label=desired_label,
                                                         use_similarity_proba=use_similarity_proba,
                                                         use_unk=use_unk, sample_proba=sample_proba,
                                                         temperature=temperature, top_n=top_n, **kwargs)

        # get max perturbed sample sentence length
        # needed to set dtype of array later and ensure the full text is used
        total_len = 0
        for word in words:
            if use_unk:
                max_len = max(3, len(word))  # len('UNK') = 3
            else:
                self.neighbors.neighbors(word)
                similar_words = self.neighbors.n[word]
                max_len = 0
                for similar_word in similar_words:
                    max_len = max(max_len, len(similar_word[0]))
            total_len += max_len + 1
        data_type = '<U' + str(int(total_len))

        # get anchors and add metadata
        exp = AnchorBaseBeam.anchor_beam(sample_fn, delta=delta,
                                         epsilon=tau, batch_size=batch_size,
                                         desired_confidence=threshold,
                                         stop_on_first=True, data_type=data_type,
                                         **kwargs)  # type: Any

        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = self.predict_fn([text])[0]
        exp = AnchorExplanation('text', exp)

        # output explanation dictionary
        explanation = {}
        explanation['names'] = exp.names()
        explanation['precision'] = exp.precision()
        explanation['coverage'] = exp.coverage()
        explanation['raw'] = exp.exp_map
        return explanation

    def perturb_sentence(self, text: str, present: list, n: int, sample_proba: float = 0.5,
                         top_n: int = 100, forbidden: set = set(), forbidden_tags: set = set(['PRP$']),
                         forbidden_words: set = set(['be']),
                         pos: set = set(['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']),
                         use_similarity_proba: bool = True, temperature: float = 1.,
                         **kwargs) -> Tuple[list, np.ndarray]:
        """
        Perturb the text instance to be explained.

        Parameters
        ----------
        text
            Text instance to be explained
        present
            List with the word index in the text for the words in the proposed anchor
        n
            Number of samples used when sampling from the corpus
        sample_proba
            Sample probability for a word if use_similarity_proba is False
        top_n
            Keep only top_n instances from similar words in the corpus
        forbidden
            Forbidden lemmas
        forbidden_tags
            Forbidden POS tags
        forbidden_words
            Forbidden words
        pos
            POS that can be changed during perturbation
        use_similarity_proba
            Bool whether to sample according to a similarity score with the corpus embeddings
        temperature
            Sample weight hyperparameter if use_similarity_proba equals True

        Returns
        -------
        raw_data
            List with num_samples of perturbed text instance
        data
            Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample
        """
        tokens = self.neighbors.nlp(text)  # get spaCy tokens for the text

        raw = np.zeros((n, len(tokens)), '|S80')
        data = np.ones((n, len(tokens)))
        raw[:] = [x.text for x in tokens]  # fill each row of the raw data matrix with the text to be explained

        for i, t in enumerate(tokens):  # apply sampling to each token

            if i in present:  # if the word is part of the anchor, move on to next token
                continue

            # check that token does not fall in any forbidden category
            if (t.text not in forbidden_words and t.pos_ in pos and
                    t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):

                # get list of tuples for neighbors with same POS tag
                # tuple = (similar word, similarity score)
                r_neighbors = [(x[0].text.encode('utf-8'), x[1])
                               for x in self.neighbors.neighbors(t.text) if x[0].tag_ == t.tag_][:top_n]

                if not r_neighbors:  # if no neighbors found with same tag, move on to next token
                    continue

                t_neighbors = [x[0] for x in r_neighbors]  # words of neighbors

                # idx for changed words with sample_proba
                n_changed = np.random.binomial(n, sample_proba)
                changed = np.random.choice(n, n_changed, replace=False)

                # check if token present in the neighbors and set weight to 0
                if t.text.encode('utf-8') in t_neighbors:
                    idx = t_neighbors.index(t.text.encode('utf-8'))
                else:
                    idx = None

                if use_similarity_proba:  # use similarity scores to sample changed tokens
                    weights = np.array([x[1] for x in r_neighbors])  # similarity scores of neighbors
                    if idx is not None:
                        weights[idx] = 0
                    weights = weights ** (1. / temperature)  # weighting by temperature
                    weights = weights / sum(weights)
                else:
                    weights = np.ones((len(r_neighbors), ))
                    if idx is not None:
                        weights /= (len(r_neighbors) - 1)
                        weights[idx] = 0
                    else:
                        weights /= len(r_neighbors)

                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights, replace=True)
                data[changed, i] = 0

        # convert numpy array into list
        raw = [' '.join([y.decode() for y in x]) for x in raw]
        return raw, data

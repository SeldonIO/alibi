import numpy as np

import spacy
import ray

from timeit import default_timer as timer

from alibi.utils.distributed import ActorPool
from alibi.utils.download import spacy_model


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.t_elapsed = timer() - self.start


class Neighbors(object):

    def __init__(self, model_name, n_similar: int = 500, w_prob: float = -15., top_n=20) -> None:
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

        self.nlp = spacy.load(model_name)
        self.w_prob = w_prob
        # list with spaCy lexemes in vocabulary
        self.to_check = [w for w in self.nlp.vocab if w.prob >= self.w_prob and w.has_vector]
        self.n_similar = n_similar
        self.top_n = top_n

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
        A list containing tuples with the similar words and similarity scores.
        """

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


class SynonymFinder(object):

    def __int__(self, text, tokenizer, model_name, n_synonyms):

        self.text = text
        self.tokenizer = tokenizer
        self.nlp_model = None
        self.synonyms = {}
        self.tokenize(text)
        self.n_synonyms = n_synonyms

    def tokenize(self, text):
        processed = self.nlp(text)                   # spaCy tokens for text
        self.words = [x.text for x in processed]     # list with words in text
        self.positions = [x.idx for x in processed]  # positions of words in text
        self.tokens = processed

    def _init_nlp_model(self):
        self.nlp_model = Neighbors(self.tokenizer, self.n_synonyms)

    def find_synonyms(self):
        if not self.nlp_model:
            self._init_nlp_model()

        for word, token in zip(self.words, self.tokens):
            if word not in self.synonyms:
                self.synonyms[word] = self.nlp_model([(word, token.tag_)])


class DistributedSynonymFinder(SynonymFinder):

    def __int__(self, *args, ncpu=2, chunksize=1):

        super(DistributedSynonymFinder, self).__init__(args)
        ray.init()
        self.nlp_model = ray.remote(Neighbors)
        self.pool = ActorPool([self.nlp_model.remote(self.model_name, top_n=self.n_synonyms) for _ in range(ncpu)])
        self.sample_fcn = lambda actor, word_tag: actor.__call__.remote(word_tag)
        self.chunksize = chunksize

    def find_synonyms(self):
        inputs = [(word, token.tag_) for word, token in zip(self.words, self.tokens)]
        neighbours = self.pool.map_unordered(
            self.sample_fcn,
            inputs,
            chunksize=self.chunksize,
        )
        for batch in neighbours:
            for result in batch:
                word = result['word']
                self.crappy_neighbours[word] = result


def main():

    sentence = 'a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification .'
    model_name = 'en_core_web_md'
    spacy_model(model=model_name)
    tokenizer = spacy.load(model_name)
    n_synonyms = 20
    finder = SynonymFinder(sentence, tokenizer, model_name, n_synonyms)
    with Timer as t:
        synonyms = finder.find_synonyms()
    print(f"Took {t.t_elapsed} to find {n_synonyms} synonyms using serial implementation!")
    ncpu, chunksize = 3, 1
    parallel_finder = DistributedSynonymFinder(sentence,
                                               tokenizer,
                                               model_name,
                                               n_synonyms,
                                               ncpu=ncpu,
                                               chunksize=chunksize,
                                               )
    with Timer as t:
        synonyms = parallel_finder.find_synonyms()
    print(f"Took {t.t_elapsed} to find {n_synonyms} synonyms using paralell implementation"
          f"with {ncpu} cores and a batch size of {chunksize}!")


if __name__ == '__main__':
    main()





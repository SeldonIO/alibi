import abc
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorflow as tf  # type: ignore
import transformers  # type: ignore
from transformers import TFAutoModelForMaskedLM, AutoTokenizer


class LanguageModel(abc.ABC):
    SUBWORD_PREFIX = ''

    def __init__(self, model_path: str, preloading: bool = True):
        """
        Initialize the language model.

        Parameters
        ----------
        model_path
            `transformers` package model path.
        preloading
            Whether to preload the online version of the transformer.
            If `False`, a call to `load` method is expected.
        """
        self.model_path = model_path
        self.model, self.caller, self.tokenizer = None, None, None

        if preloading:
            # set model
            self.model = TFAutoModelForMaskedLM.from_pretrained(model_path)
            self.caller = tf.function(self.model.call, experimental_relax_shapes=True)

            # set tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def from_disk(self, path: Union[str, Path]):
        """
        Loads a model from disk.

        Parameters
        ----------
        path
            Path to the checkpoint.
        """
        # set model
        self.model = TFAutoModelForMaskedLM.from_pretrained(path, local_files_only=True)
        self.caller = tf.function(self.model.call, experimental_relax_shapes=True)

        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def to_disk(self, path: Union[str, Path]):
        """
        Saves a model to disk.

        Parameters
        ----------
        path
            Path to the checkpoint.
        """
        # save model if set
        if self.model:
            self.model.save_pretrained(path)

        # save tokenizer if set
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    @abc.abstractmethod
    def is_subword_prefix(self, token: str) -> bool:
        """
        Checks if the given token in a subword.

        Parameters
        ----------
        token
            Token to be checked if it is a subword.

        Returns
        -------
        True if the given token is a subword. False otherwise.
        """
        pass

    def select_entire_word(self, text: List[str], start_idx: int, punctuation: str) -> str:
        """
        Given a text and the starting index of a word, the function selects the entire word.

        Parameters
        ----------
        text
            Full text.
        start_idx
            Starting index of a word.
        punctuation
            String of punctuation to be considered. If it encounters a token
             composed only of characters in `punctuation` it terminates the search.

        Returns
        -------
        The entire words (this includes the subwords that come after).
        """
        # define the ending index
        end_idx = start_idx + 1

        while end_idx < len(text):
            # The second condition is necessary for models like Roberta.
            # If the second condition is not included, it can select words like: `word,` instead of `word`
            if (not self.is_subword_prefix(text[end_idx])) or self.is_punctuation(text[end_idx], punctuation):
                break

            end_idx += 1

        # convert the tokens into a string
        word = self.tokenizer.convert_tokens_to_string(text[start_idx:end_idx])
        return word

    def is_stop_word(self, text: List[str], start_idx: int, punctuation: str, stopwords: Optional[List[str]]) -> bool:
        """
        Checks if the given words starting at the given index is in the list of stopwords.

        Parameters
        ----------
        text
            Full text.
        start_idx
            Starting index of a word.
        stopwords:
            List of stop words. The words in this list should be lowercase.
        punctuation
            Punctuation to be considered. See `select_entire_word`.

        Returns
        -------
        True if the `token` is in the `stopwords` list. False otherwise.
        """
        if not stopwords:
            return False

        if self.is_subword_prefix(text[start_idx]):
            return False

        word = self.select_entire_word(text, start_idx=start_idx, punctuation=punctuation).strip()
        return word.lower() in stopwords

    def is_punctuation(self, token: str, punctuation: str) -> bool:
        """
        Checks if the given token is punctuation.

        Parameters
        ----------
        token
            Token to be checked if it is punctuation.
        punctuation
            String containing all punctuation to be considered.

        Returns
        -------
        True if the `token` is a punctuation. False otherwise.
        """
        token = token.replace(self.SUBWORD_PREFIX, '').strip()
        return all([c in punctuation for c in token])

    @property
    @abc.abstractmethod
    def mask(self) -> str:
        """
        Returns the mask token.
        """
        pass

    @property
    def mask_token(self) -> int:
        """
        Returns the mask token id
        """
        return self.tokenizer.mask_token_id

    @property
    def max_num_tokens(self) -> int:
        """
        Returns the maximum number of token allowed by the model.
        """
        return self.model.config.max_position_embeddings

    def head_tail_split(self, text: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Split the text in head and tail. Some language models support a maximum
        number of tokens. Thus is necessary to split the text to meet this constraint.
        After the text is split in head and tail, only the head is considered for operation.
        Thus the tail will remain unchanged.

        Parameters
        ----------
        text
            Text to be split in head and tail.

        Returns
        -------
        Tuple consisting of the head, tail and their corresponding list of tokens.
        """
        text = text.strip()
        if len(text) == 0:
            raise ValueError("The text is empty")

        # data = `This is not a wordy sentence` -> tokens = [this, is, not, a, word, ##y, sentence, .]
        tokens: List[str] = self.tokenizer.tokenize(text)

        # some models do not have a max length restrictions (e.g. XLNet)
        if self.max_num_tokens == -1 or len(tokens) <= self.max_num_tokens:
            return text, '', tokens, []

        # head's length
        head_num_tokens = self.max_num_tokens

        # decrease the head length so it contains full words
        while (head_num_tokens > 0) and self.is_subword_prefix(tokens[head_num_tokens]):
            head_num_tokens -= 1

        if head_num_tokens == 0:
            raise ValueError("Check the first word in the sentence. Seems it is a very long word")

        ids = self.tokenizer.convert_tokens_to_ids(tokens[:head_num_tokens])
        head_text = self.tokenizer.decode(ids).strip()
        tail_text = ''

        # if the number of tokens exceeds the maximum allowed
        # number, then construct also the tail_text
        if len(tokens) >= head_num_tokens:
            ids = self.tokenizer.convert_tokens_to_ids(tokens[head_num_tokens:])
            tail_text = self.tokenizer.decode(ids).strip()

        return head_text, tail_text, tokens[:head_num_tokens], tokens[head_num_tokens:]

    @tf.autograph.experimental.do_not_convert
    def predict_batch_lm(self,
                         x: transformers.tokenization_utils_base.BatchEncoding,
                         vocab_size: int,
                         batch_size: int) -> np.ndarray:
        """
        Tensorflow language model batch predictions for AnchorText.

        Parameters
        ----------
        x
            Batch of instances.
        vocab_size
            Vocabulary size of language model.
        batch_size
            Batch size used for predictions.

        Returns
        -------
        y
            Array with model predictions.
        """
        n, m = x['input_ids'].shape
        y = np.zeros((n, m, vocab_size), dtype=np.float32)
        n_minibatch = int(np.ceil(n / batch_size))

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = dict()

            if 'input_ids' in x.keys():
                x_batch['input_ids'] = x['input_ids'][istart:istop]

            if 'token_type_ids' in x.keys():
                x_batch['token_type_ids'] = x['token_type_ids'][istart:istop]

            if 'attention_mask' in x.keys():
                x_batch['attention_mask'] = x['attention_mask'][istart:istop]

            y[istart:istop] = self.caller(**x_batch)[0].numpy()
        return y


class DistilbertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, preloading: bool = True):
        """
        Initialize DistilbertBaseUncased.

        Parameters
        ----------
        preloading
            See `LanguageModel` constructor.
        """
        super(DistilbertBaseUncased, self).__init__("distilbert-base-uncased", preloading)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return DistilbertBaseUncased.SUBWORD_PREFIX in token


class BertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, preloading: bool = True):
        """
        Initialize BertBaseUncased.

        Parameters
        ----------
        preloading
            See `LanguageModel` constructor.
        """
        super(BertBaseUncased, self).__init__("bert-base-uncased", preloading)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return BertBaseUncased.SUBWORD_PREFIX in token


class RobertaBase(LanguageModel):
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, preloading: bool = True):
        """
        Initialize RobertaBase

        Parameters
        ----------
        preloading
            See `LanguageModel` constructor.
        """
        super(RobertaBase, self).__init__("roberta-base", preloading)

    @property
    def mask(self) -> str:
        return RobertaBase.SUBWORD_PREFIX + self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return RobertaBase.SUBWORD_PREFIX not in token

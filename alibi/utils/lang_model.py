"""
This module defines a wrapper for transformer-based masked language models used in `AnchorText` as a perturbation
strategy. The `LanguageModel` base class defines basic functionalities as loading, storing, and predicting.

Language model's tokenizers usually work at a subword level, and thus, a word can be split into subwords. For example,
a word can be decomposed as: ``word = [head_token tail_token_1 tail_token_2 ... tail_token_k]``. For language models
such as `DistilbertBaseUncased` and `BertBaseUncased`, the tail tokens can be identified by a special prefix ``'##'``.
On the other hand, for `RobertaBase` only the head is prefixed with the special character ``'Ġ'``, thus the tail tokens
can be identified by the absence of the special token. In this module, we refer to a tail token as a subword prefix.
We will use the notion of a subword to refer to either a `head` or a `tail` token.

To generate interpretable perturbed instances, we do not mask subwords, but entire words. Note that this operation is
equivalent to replacing the head token with the special mask token, and removing the tail tokens if they exist. Thus,
the `LanguageModel` class offers additional functionalities such as: checking if a token is a subword prefix,
selection of a word (head_token along with the tail_tokens), etc.

Some language models can work with a limited number of tokens, thus the input text has to be split. Thus, a text will
be split in head and tail, where the number of tokens in the head is less or equal to the maximum allowed number of
tokens to be processed by the language model. In the `AnchorText` only the head is perturbed. To keep the results
interpretable, we ensure that the head will not end with a subword, and will contain only full words.
"""

import abc
import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import tensorflow as tf
import transformers
from transformers import TFAutoModelForMaskedLM, AutoTokenizer


class LanguageModel(abc.ABC):
    SUBWORD_PREFIX = ''  #: Language model subword prefix.

    # We don't type transformers objects here as it would likely require digging into
    # some private base classes which may change in the future and cause breaking changes.
    model: Any
    tokenizer: Any
    # TODO: from TF 2.6 this has type `tf.types.experimental.GenericFunction`,
    #  unsure if we can be more specific right now
    caller: Callable

    def __init__(self, model_path: str, preloading: bool = True):
        """
        Initialize the language model.

        Parameters
        ----------
        model_path
            `transformers` package model path.
        preloading
            Whether to preload the online version of the transformer. If ``False``, a call to `from_disk`
            method is expected.
        """
        self.model_path = model_path

        if preloading:
            # set model (for performance reasons the `call` method is wrapped in tf.function)
            self.model = TFAutoModelForMaskedLM.from_pretrained(model_path)

            # To understand the type: ignore see https://github.com/python/mypy/issues/2427
            self.caller = tf.function(self.model.call, experimental_relax_shapes=True)  # type: ignore[assignment]

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
        # set model (for performance reasons the `call` method is wrapped in tf.function)
        self.model = TFAutoModelForMaskedLM.from_pretrained(path, local_files_only=True)

        # To understand the type: ignore see https://github.com/python/mypy/issues/2427
        self.caller = tf.function(self.model.call, experimental_relax_shapes=True)  # type: ignore[assignment]

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
        Checks if the given token is a part of the tail of a word. Note that a word can
        be split in multiple tokens (e.g., ``word = [head_token tail_token_1 tail_token_2 ... tail_token_k]``).
        Each language model has a convention on how to mark a tail token. For example
        `DistilbertBaseUncased` and `BertBaseUncased` have the tail tokens prefixed with the special
        set of characters ``'##'``. On the other hand, for `RobertaBase` only the head token is prefixed
        with the special character ``'Ġ'`` and thus we need to check the absence of the prefix to identify
        the tail tokens. We call those special characters `SUBWORD_PREFIX`. Due to different conventions,
        this method has to be implemented for each language model. See module docstring for namings.

        Parameters
        ----------
        token
            Token to be checked if it is a subword.

        Returns
        -------
        ``True`` if the given token is a subword prefix. ``False`` otherwise.
        """
        pass

    def select_word(self,
                    tokenized_text: List[str],
                    start_idx: int,
                    punctuation: str) -> str:
        """
        Given a tokenized text and the starting index of a word, the function selects the entire word.
        Note that a word is composed of multiple tokens (e.g., ``word = [head_token tail_token_1
        tail_token_2 ... tail_token_k]``). The tail tokens can be identified based on the
        presence/absence of `SUBWORD_PREFIX`. See :py:meth:`alibi.utils.lang_model.LanguageModel.is_subword_prefix`
        for more details.

        Parameters
        ----------
        tokenized_text
            Tokenized text.
        start_idx
            Starting index of a word.
        punctuation
            String of punctuation to be considered. If it encounters a token
            composed only of characters in `punctuation` it terminates the search.

        Returns
        -------
        The word obtained by concatenation ``[head_token tail_token_1 tail_token_2 ... tail_token_k]``.
        """
        # define the ending index
        end_idx = start_idx + 1

        while end_idx < len(tokenized_text):
            # The second condition is necessary for models like Roberta.
            # If the second condition is not included, it can select words like: `word,` instead of `word`
            if (not self.is_subword_prefix(tokenized_text[end_idx])) or \
                    self.is_punctuation(tokenized_text[end_idx], punctuation):
                break

            end_idx += 1

        # convert the tokens into a string
        word = self.tokenizer.convert_tokens_to_string(tokenized_text[start_idx:end_idx])
        return word

    def is_stop_word(self,
                     tokenized_text: List[str],
                     start_idx: int,
                     punctuation: str,
                     stopwords: Optional[List[str]]) -> bool:
        """
        Checks if the given word starting at the given index is in the list of stopwords.

        Parameters
        ----------
        tokenized_text
            Tokenized text.
        start_idx
            Starting index of a word.
        stopwords:
            List of stop words. The words in this list should be lowercase.
        punctuation
            Punctuation to be considered. See :py:meth:`alibi.utils.lang_model.LanguageModel.select_entire_word`.

        Returns
        -------
        ``True`` if the `token` is in the `stopwords` list. ``False`` otherwise.
        """
        if not stopwords:
            return False

        if self.is_subword_prefix(tokenized_text[start_idx]):
            return False

        word = self.select_word(tokenized_text, start_idx=start_idx, punctuation=punctuation).strip()
        return word.strip().lower() in stopwords

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
        ``True`` if the `token` is a punctuation. ``False`` otherwise.
        """
        # need to convert tokens from transformers encoding representation into unicode to allow for proper
        # punctuation check against a list of punctuation unicode characters provided (e.g., for RobertaBase this will
        # convert special  characters such as 'Ġ[âĢ¦]' into ' […]', and will allow us to check if characters such
        # as ' ', '[', '…', ']' appear in the punctuation list.
        string_rep = self.tokenizer.convert_tokens_to_string([token]).strip()

        # have to remove `##` prefix for Bert & DistilBert to allow the check with `any`
        if string_rep.startswith(self.SUBWORD_PREFIX):
            string_rep = string_rep.replace(self.SUBWORD_PREFIX, '', 1)

        return any([c in punctuation for c in string_rep]) if len(string_rep) else False

    @property
    @abc.abstractmethod
    def mask(self) -> str:
        """
        Returns the mask token.
        """
        pass

    @property
    def mask_id(self) -> int:
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
            raise ValueError("The text is empty.")

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

    def predict_batch_lm(self,
                         x: transformers.tokenization_utils_base.BatchEncoding,
                         vocab_size: int,
                         batch_size: int) -> np.ndarray:
        """
        `Tensorflow` language model batch predictions for `AnchorText`.

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

            y[istart:istop] = self.caller(**x_batch).logits.numpy()
        return y


class DistilbertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, preloading: bool = True):
        """
        Initialize `DistilbertBaseUncased`.

        Parameters
        ----------
        preloading
            See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__`.
        """
        super().__init__("distilbert-base-uncased", preloading)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return token.startswith(DistilbertBaseUncased.SUBWORD_PREFIX)


class BertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, preloading: bool = True):
        """
        Initialize `BertBaseUncased`.

        Parameters
        ----------
        preloading
            See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__`.
        """
        super().__init__("bert-base-uncased", preloading)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return token.startswith(BertBaseUncased.SUBWORD_PREFIX)


class RobertaBase(LanguageModel):
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, preloading: bool = True):
        """
        Initialize `RobertaBase`.

        Parameters
        ----------
        preloading
            See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__` constructor.
        """
        super().__init__("roberta-base", preloading)

    @property
    def mask(self) -> str:
        return RobertaBase.SUBWORD_PREFIX + self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return not token.startswith(RobertaBase.SUBWORD_PREFIX)

import abc
import numpy as np
from typing import List
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextTransformer(abc.ABC):

    @abc.abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """
        Transform a batch of raw text strings into a list of lists of string tokens.

        Parameters
        ----------
        texts
            List of text strings.

        Returns
        -------
        List of lists of string tokens.

        """
        pass

    @abc.abstractmethod
    def texts_to_array(self, texts: List[str]) -> np.ndarray:
        """
        Transform a batch of raw text strings into a homogenous numpy array for feeding into a model.

        Parameters
        ----------
        texts
            List of text strings.
        kwargs
            Custom keyword arguments.

        Returns
        -------
        A homogenous numpy array to be sent to the model.

        """
        pass

    def array_to_ragged_array(self, array: np.ndarray, value: float = 0.0) -> List[np.ndarray]:
        """
        Transform a homogenous numpy array into a ragged list of arrays by removing entries with `value`
        representing the padding token.

        Parameters
        ----------
        array
            Homogenous numpy array.
        value
            Value represneting the padding token.

        Returns
        -------
        List of heterogenous numpy arrays.

        """
        list_arrays = []
        for row in array:
            list_arrays.append(
                row[row != value].tolist())  # TODO: assumed here `value` is not a meaningful token value.
        return list_arrays


class KerasTextTransformer(TextTransformer):

    def __init__(self, tokenizer: tf.keras.preprocessing.text.Tokenizer, maxlen: int) -> None:
        """
        Initialize a text transformer based on a Keras text tokenizer.

        Parameters
        ----------
        tokenizer
            A fitted Keras text tokenizer instance.
        maxlen
            Maximum length of the homogenous array the raw text is transformed into. Texts with
            a number of tokens smaller than this will be zero-padded and texts longer than this
            will be truncated. This can be customized by inheriting and overriding the `texts_to_array`
            method.

        """
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def texts_to_tokens(self, texts):
        tokenized_texts = []
        seq_texts = self.tokenizer.texts_to_sequences(texts)
        for text in seq_texts:
            tokens = [self.tokenizer.index_word[ix] for ix in text]
            tokenized_texts.append(tokens)

        return tokenized_texts

    def texts_to_array(self, texts):
        return pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=self.maxlen)

import string
from functools import partial

from typing import (Dict, List, Optional, Tuple, Type)

import numpy as np
import tensorflow as tf

from alibi.utils.lang_model import LanguageModel
from alibi.explainers.anchors.text_samplers import AnchorTextSampler


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
                continue

            # Add punctuation in the sampling mask. This means that the
            # punctuation will not be considered when sampling for the masked words.
            sample_punctuation: bool = perturb_opts.get('sample_punctuation', False)
            punctuation: str = perturb_opts.get('punctuation', string.punctuation)

            if (not sample_punctuation) and self.model.is_punctuation(token, punctuation):
                self.subwords_mask[vocab[token]] = True

        # define head, tail part of the text
        self.head: str = ''
        self.tail: str = ''
        self.head_tokens: List[str] = []
        self.tail_tokens: List[str] = []

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
        See :py:meth:`alibi.explainers.anchors.language_model_text_sampler.LanguageModelSampler.perturb_sentence`.
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
        data = np.ones((mask_templates, len(self.ids_sample)), dtype=np.int32)
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
            tokens_plus['input_ids'] = tf.convert_to_tensor(tokens)
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

    def seed(self, seed: int) -> None:
        tf.random.set_seed(seed)

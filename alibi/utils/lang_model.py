import abc
import string
import numpy as np
from typing import List, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer


class LanguageModel(abc.ABC):
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the language model.

        Parameters
        ----------
        model_path
            `transformers` package model path.
        device
            Device to use: cpu or cuda.
        """
        self.model_path = model_path
        self.device = torch.device(device)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # load and send model to device
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

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
        Given a text and the starting index of a word, the function
        selects the entier word.

        Parameters
        ----------
        text
            Full text.
        start_idx
            Starting index of a word.

        Returns
        -------
        The entire words (this includes the subwords that come after).
        """
        # define the ending index
        end_idx = start_idx + 1

        while (end_idx < len(text)):
            # The second condition is necessary for models like Roberta.
            # If the second condition is not included, it can select words like: `word,` instaed of `word`
            if (not self.is_subword_prefix(text[end_idx])) or self.is_punctuation(text[end_idx], punctuation):
                break

            end_idx += 1

        # convert the tokens into a string
        word = self.tokenizer.convert_tokens_to_string(text[start_idx:end_idx])
        return word

    def is_stop_word(self, text: List[str], start_idx: int, punctuation: str, stopwords: Optional[List[str]]) -> bool:
        """
        Checks if the given words starting at the given index is in the list of stopwords

        Parameters
        ----------
        text
            Full text.
        start_idx
            Starting index of a word.
        stopwords:
            List of stop words. The words in this list should be lowercase.

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
        return all([chr in punctuation for chr in token])

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

    def head_tail_split(self, text: str) -> Tuple[str, Optional[str], List[str], Optional[List[str]]]:
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
            return text, None, tokens, []

        # head's length
        head_num_tokens = self.max_num_tokens

        # decrease the head length so it contains full words
        while (head_num_tokens > 0) and self.is_subword_prefix(tokens[head_num_tokens]):
            head_num_tokens -= 1

        if head_num_tokens == 0:
            raise ValueError("Check the first word in the sentence. Seems it is a very long word")

        ids = self.tokenizer.convert_tokens_to_ids(tokens[:head_num_tokens])
        head_text = self.tokenizer.decode(ids).strip()
        tail_text = None

        # if the number of tokens exceeds the maximum allowed
        # number, then construct also the tail_text
        if len(tokens) >= head_num_tokens:
            ids = self.tokenizer.convert_tokens_to_ids(tokens[head_num_tokens:])
            tail_text = self.tokenizer.decode(ids).strip()

        return head_text, tail_text, tokens[:head_num_tokens], tokens[head_num_tokens:]

    def predict_batch_lm(self,
                         x: transformers.tokenization_utils_base.BatchEncoding,
                         device: torch.device,
                         vocab_size: int,
                         batch_size: int) -> torch.Tensor:
        """
        PyTorch language model batch predictions for AnchorText.

        Parameters
        ----------
        x
            Batch of instances.
        device
            Device used for the model.
        vocab_size
            Vocabulary size of language model.
        batch_size
            Batch size used for predictions.

        Returns
        -------
        y
            Tensor with model predictions.
        """
        is_cuda = device.type == 'cuda'
        n, m = x['input_ids'].shape
        y = torch.zeros((n, m, vocab_size), dtype=torch.float32)
        n_minibatch = int(np.ceil(n / batch_size))

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = dict()

            if 'input_ids' in x.keys():
                x_batch['input_ids'] = x['input_ids'][istart:istop].to(device)

            if 'token_type_ids' in x.keys():
                x_batch['token_type_ids'] = x['token_type_ids'][istart:istop].to(device)

            if 'attention_mask' in x.keys():
                x_batch['attention_mask'] = x['attention_mask'][istart:istop].to(device)

            preds = self.model(**x_batch)[0]
            y[istart:istop] = preds.cpu().detach() if is_cuda else preds.detach()
        return y


class DistilbertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, device: str = "cuda"):
        super(DistilbertBaseUncased, self).__init__("distilbert-base-uncased", device)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return DistilbertBaseUncased.SUBWORD_PREFIX in token


class BertBaseUncased(LanguageModel):
    SUBWORD_PREFIX = '##'

    def __init__(self, device: str = "cuda"):
        super(BertBaseUncased, self).__init__("bert-base-uncased", device)

    @property
    def mask(self) -> str:
        return self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return BertBaseUncased.SUBWORD_PREFIX in token


class RobertaBase(LanguageModel):
    SUBWORD_PREFIX = 'Ġ'

    def __init__(self, device: str = "cuda"):
        super(RobertaBase, self).__init__("roberta-base", device)

    @property
    def mask(self):
        return RobertaBase.SUBWORD_PREFIX + self.tokenizer.mask_token

    def is_subword_prefix(self, token: str) -> bool:
        return RobertaBase.SUBWORD_PREFIX not in token


# def test_functionalities(lm: LanguageModel, text):
#     stopwords = ['and', 'the', 'but', 'a', 'this']
#
#     tokens = lm.tokenizer.tokenize(text)
#     string_tokens = lm.tokenizer.convert_tokens_to_string(tokens)
#
#     print("Tokens:", tokens)
#     print("String:", string_tokens)
#     print("Ids:", lm.tokenizer.convert_tokens_to_ids(tokens))
#
#     stopwords = [token for i, token in enumerate(tokens) if lm.is_stop_word(
#                                                             text=tokens,
#                                                             start_idx=i,
#                                                             punctuation=string.punctuation,
#                                                             stopwords=stopwords)]
#     print("Stopwords:", stopwords)
#
#     punctuation = [token for token in tokens if lm.is_punctuation(token, string.punctuation)]
#     print("Punctuation:", punctuation)
#
#
# if __name__ == "__main__":
#     text = "this and this this is a first sentence !?! but nothing afterwards don't matter..."
#
#     print("\n\n RobertaBase \n =============== \n")
#     lm = RobertaBase()
#     test_functionalities(lm, text)
#     del lm
#
#     print("\n\n BertBaseUncased \n =============== \n")
#     lm = BertBaseUncased()
#     test_functionalities(lm, text)
#     del lm
#
#     print("\n\n DistilbertBaseUncased \n =============== \n")
#     lm = DistilbertBaseUncased()
#     test_functionalities(lm, text)
#     del lm





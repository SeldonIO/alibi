# `alibi.utils.lang_model`

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

## `BertBaseUncased`

_Inherits from:_ `LanguageModel`, `ABC`

### Constructor

```python
BertBaseUncased(self, preloading: bool = True)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `preloading` | `bool` | `True` | See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__`. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `mask` | `str` |  |

### Methods

#### `is_subword_prefix`

```python
is_subword_prefix(token: str) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `token` | `str` |  | Token to be checked if it is a subword. |

**Returns**
- Type: `bool`

## `DistilbertBaseUncased`

_Inherits from:_ `LanguageModel`, `ABC`

### Constructor

```python
DistilbertBaseUncased(self, preloading: bool = True)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `preloading` | `bool` | `True` | See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__`. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `mask` | `str` |  |

### Methods

#### `is_subword_prefix`

```python
is_subword_prefix(token: str) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `token` | `str` |  | Token to be checked if it is a subword. |

**Returns**
- Type: `bool`

## `LanguageModel`

_Inherits from:_ `ABC`

### Constructor

```python
LanguageModel(self, model_path: str, preloading: bool = True)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model_path` | `str` |  | `transformers` package model path. |
| `preloading` | `bool` | `True` | Whether to preload the online version of the transformer. If ``False``, a call to `from_disk` method is expected. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `mask` | `str` | Returns the mask token. |
| `mask_id` | `int` | Returns the mask token id |
| `max_num_tokens` | `int` | Returns the maximum number of token allowed by the model. |

### Methods

#### `from_disk`

```python
from_disk(path: Union[str, pathlib.Path])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, pathlib.Path]` |  | Path to the checkpoint. |

#### `head_tail_split`

```python
head_tail_split(text: str) -> Tuple[str, str, List[str], List[str]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  | Text to be split in head and tail. |

**Returns**
- Type: `Tuple[str, str, List[str], List[str]]`

#### `is_punctuation`

```python
is_punctuation(token: str, punctuation: str) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `token` | `str` |  | Token to be checked if it is punctuation. |
| `punctuation` | `str` |  | String containing all punctuation to be considered. |

**Returns**
- Type: `bool`

#### `is_stop_word`

```python
is_stop_word(tokenized_text: List[str], start_idx: int, punctuation: str, stopwords: Optional[List[str]]) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `tokenized_text` | `List[str]` |  | Tokenized text. |
| `start_idx` | `int` |  | Starting index of a word. |
| `punctuation` | `str` |  | Punctuation to be considered. See :py:meth:`alibi.utils.lang_model.LanguageModel.select_entire_word`. |
| `stopwords` | `Optional[List[str]]` |  | List of stop words. The words in this list should be lowercase. |

**Returns**
- Type: `bool`

#### `is_subword_prefix`

```python
is_subword_prefix(token: str) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `token` | `str` |  | Token to be checked if it is a subword. |

**Returns**
- Type: `bool`

#### `predict_batch_lm`

```python
predict_batch_lm(x: transformers.tokenization_utils_base.BatchEncoding, vocab_size: int, batch_size: int) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `transformers.tokenization_utils_base.BatchEncoding` |  | Batch of instances. |
| `vocab_size` | `int` |  | Vocabulary size of language model. |
| `batch_size` | `int` |  | Batch size used for predictions. |

**Returns**
- Type: `numpy.ndarray`

#### `select_word`

```python
select_word(tokenized_text: List[str], start_idx: int, punctuation: str) -> str
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `tokenized_text` | `List[str]` |  | Tokenized text. |
| `start_idx` | `int` |  | Starting index of a word. |
| `punctuation` | `str` |  | String of punctuation to be considered. If it encounters a token composed only of characters in `punctuation` it terminates the search. |

**Returns**
- Type: `str`

#### `to_disk`

```python
to_disk(path: Union[str, pathlib.Path])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, pathlib.Path]` |  | Path to the checkpoint. |

## `RobertaBase`

_Inherits from:_ `LanguageModel`, `ABC`

### Constructor

```python
RobertaBase(self, preloading: bool = True)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `preloading` | `bool` | `True` | See :py:meth:`alibi.utils.lang_model.LanguageModel.__init__` constructor. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `mask` | `str` |  |

### Methods

#### `is_subword_prefix`

```python
is_subword_prefix(token: str) -> bool
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `token` | `str` |  | Token to be checked if it is a subword. |

**Returns**
- Type: `bool`

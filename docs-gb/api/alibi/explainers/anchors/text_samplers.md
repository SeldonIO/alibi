# `alibi.explainers.anchors.text_samplers`
## Classes
### `AnchorTextSampler`

#### Constructor

```python
AnchorTextSampler(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

#### Methods

##### `set_text`

```python
set_text(text: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

### `Neighbors`

#### Constructor

```python
Neighbors(self, nlp_obj: 'spacy.language.Language', n_similar: int = 500, w_prob: float = -15.0) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp_obj` | `spacy.language.Language` |  |  |
| `n_similar` | `int` | `500` |  |
| `w_prob` | `float` | `-15.0` |  |

#### Methods

##### `neighbors`

```python
neighbors(word: str, tag: str, top_n: int) -> dict
```

Find similar words for a certain word in the vocabulary.

Parameters
----------
word
    Word for which we need to find similar words.
tag
    Part of speech tag for the words.
top_n
    Return only `top_n` neighbors.

Returns
-------
A dict with two fields. The ``'words'`` field contains a `numpy` array of the `top_n` most similar words,         whereas the fields ``'similarities'`` is a `numpy` array with corresponding word similarities.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `word` | `str` |  |  |
| `tag` | `str` |  |  |
| `top_n` | `int` |  |  |

**Returns**
- Type: `dict`

### `SimilaritySampler` (_inherits from `AnchorTextSampler`)

#### Constructor

```python
SimilaritySampler(self, nlp: 'spacy.language.Language', perturb_opts: Dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp` | `spacy.language.Language` |  |  |
| `perturb_opts` | `Dict` |  |  |

#### Methods

##### `find_similar_words`

```python
find_similar_words() -> None
```

This function queries a `spaCy` nlp model to find `n` similar words with the same

part of speech for each word in the instance to be explained. For each word
the search procedure returns a dictionary containing a `numpy` array of words (``'words'``)
and a `numpy` array of word similarities (``'similarities'``).

**Returns**
- Type: `None`

##### `perturb_sentence_similarity`

```python
perturb_sentence_similarity(present: tuple, n: int, sample_proba: float = 0.5, forbidden: frozenset = frozenset(), forbidden_tags: frozenset = frozenset({'PRP$'}), forbidden_words: frozenset = frozenset({'be'}), temperature: float = 1.0, pos: frozenset = frozenset({'DET', 'ADV', 'VERB', 'ADP', 'ADJ', 'NOUN'}), use_proba: bool = False, kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Perturb the text instance to be explained.

Parameters
----------
present
    Word index in the text for the words in the proposed anchor.
n
    Number of samples used when sampling from the corpus.
sample_proba
    Sample probability for a word if `use_proba=False`.
forbidden
    Forbidden lemmas.
forbidden_tags
    Forbidden POS tags.
forbidden_words
    Forbidden words.
pos
    POS that can be changed during perturbation.
use_proba
    Bool whether to sample according to a similarity score with the corpus embeddings.
temperature
    Sample weight hyper-parameter if ``use_proba=True``.
**kwargs
    Other arguments. Not used.

Returns
-------
raw
    Array of perturbed text instances.
data
    Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `present` | `tuple` |  |  |
| `n` | `int` |  |  |
| `sample_proba` | `float` | `0.5` |  |
| `forbidden` | `frozenset` | `frozenset()` |  |
| `forbidden_tags` | `frozenset` | `frozenset({'PRP$'})` |  |
| `forbidden_words` | `frozenset` | `frozenset({'be'})` |  |
| `temperature` | `float` | `1.0` |  |
| `pos` | `frozenset` | `frozenset({'DET', 'ADV', 'VERB', 'ADP', 'ADJ', 'NOUN'})` |  |
| `use_proba` | `bool` | `False` |  |
| `kwargs` |  |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

##### `set_data_type`

```python
set_data_type() -> None
```

Working with `numpy` arrays of strings requires setting the data type to avoid

truncating examples. This function estimates the longest sentence expected
during the sampling process, which is used to set the number of characters
for the samples and examples arrays. This depends on the perturbation method
used for sampling.

**Returns**
- Type: `None`

##### `set_text`

```python
set_text(text: str) -> None
```

Sets the text to be processed

Parameters
----------
text
    Text to be processed.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

### `UnknownSampler` (_inherits from `AnchorTextSampler`)

#### Constructor

```python
UnknownSampler(self, nlp: 'spacy.language.Language', perturb_opts: Dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp` | `spacy.language.Language` |  |  |
| `perturb_opts` | `Dict` |  |  |

#### Methods

##### `set_data_type`

```python
set_data_type() -> None
```

Working with `numpy` arrays of strings requires setting the data type to avoid

truncating examples. This function estimates the longest sentence expected
during the sampling process, which is used to set the number of characters
for the samples and examples arrays. This depends on the perturbation method
used for sampling.

**Returns**
- Type: `None`

##### `set_text`

```python
set_text(text: str) -> None
```

Sets the text to be processed.

Parameters
----------
text
    Text to be processed.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

## Functions
### `load_spacy_lexeme_prob`

```python
load_spacy_lexeme_prob(nlp: spacy.language.Language) -> spacy.language.Language
```

This utility function loads the `lexeme_prob` table for a spacy model if it is not present.

This is required to enable support for different spacy versions.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp` | `spacy.language.Language` |  |  |

**Returns**
- Type: `spacy.language.Language`

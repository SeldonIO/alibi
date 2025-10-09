# `alibi.explainers.anchors.text_samplers`
## `AnchorTextSampler`

### Constructor

```python
AnchorTextSampler(self, /, *args, **kwargs)
```
### Methods

#### `set_text`

```python
set_text(text: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

## `Neighbors`

### Constructor

```python
Neighbors(self, nlp_obj: 'spacy.language.Language', n_similar: int = 500, w_prob: float = -15.0) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp_obj` | `spacy.language.Language` |  |  |
| `n_similar` | `int` | `500` |  |
| `w_prob` | `float` | `-15.0` |  |

### Methods

#### `neighbors`

```python
neighbors(word: str, tag: str, top_n: int) -> dict
```

Find similar words for a certain word in the vocabulary.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `word` | `str` |  |  |
| `tag` | `str` |  |  |
| `top_n` | `int` |  |  |

**Returns**
- Type: `dict`

## `SimilaritySampler`

_Inherits from:_ `AnchorTextSampler`

### Constructor

```python
SimilaritySampler(self, nlp: 'spacy.language.Language', perturb_opts: Dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp` | `spacy.language.Language` |  |  |
| `perturb_opts` | `Dict` |  |  |

### Methods

#### `find_similar_words`

```python
find_similar_words() -> None
```

This function queries a `spaCy` nlp model to find `n` similar words with the same

part of speech for each word in the instance to be explained. For each word
the search procedure returns a dictionary containing a `numpy` array of words (``'words'``)
and a `numpy` array of word similarities (``'similarities'``).

**Returns**
- Type: `None`

#### `perturb_sentence_similarity`

```python
perturb_sentence_similarity(present: tuple, n: int, sample_proba: float = 0.5, forbidden: frozenset = frozenset(), forbidden_tags: frozenset = frozenset({'PRP$'}), forbidden_words: frozenset = frozenset({'be'}), temperature: float = 1.0, pos: frozenset = frozenset({'ADV', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ'}), use_proba: bool = False, kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Perturb the text instance to be explained.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `present` | `tuple` |  |  |
| `n` | `int` |  |  |
| `sample_proba` | `float` | `0.5` |  |
| `forbidden` | `frozenset` | `frozenset()` |  |
| `forbidden_tags` | `frozenset` | `frozenset({'PRP$'})` |  |
| `forbidden_words` | `frozenset` | `frozenset({'be'})` |  |
| `temperature` | `float` | `1.0` |  |
| `pos` | `frozenset` | `frozenset({'ADV', 'NOUN', 'VERB', 'ADP', 'DET', 'ADJ'})` |  |
| `use_proba` | `bool` | `False` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `set_data_type`

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

#### `set_text`

```python
set_text(text: str) -> None
```

Sets the text to be processed

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

## `UnknownSampler`

_Inherits from:_ `AnchorTextSampler`

### Constructor

```python
UnknownSampler(self, nlp: 'spacy.language.Language', perturb_opts: Dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `nlp` | `spacy.language.Language` |  |  |
| `perturb_opts` | `Dict` |  |  |

### Methods

#### `set_data_type`

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

#### `set_text`

```python
set_text(text: str) -> None
```

Sets the text to be processed.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  |  |

**Returns**
- Type: `None`

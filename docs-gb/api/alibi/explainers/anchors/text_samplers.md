# `alibi.explainers.anchors.text_samplers`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.anchors.text_samplers (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

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
| `nlp_obj` | `spacy.language.Language` |  | `spaCy` model. |
| `n_similar` | `int` | `500` | Number of similar words to return. |
| `w_prob` | `float` | `-15.0` | Smoothed log probability estimate of token's type. |

### Methods

#### `neighbors`

```python
neighbors(word: str, tag: str, top_n: int) -> dict
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `word` | `str` |  | Word for which we need to find similar words. |
| `tag` | `str` |  | Part of speech tag for the words. |
| `top_n` | `int` |  | Return only `top_n` neighbors. |

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
| `nlp` | `spacy.language.Language` |  | `spaCy` object. |
| `perturb_opts` | `Dict` |  | Perturbation options. |

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
perturb_sentence_similarity(present: tuple, n: int, sample_proba: float = 0.5, forbidden: frozenset = frozenset(), forbidden_tags: frozenset = frozenset({'PRP$'}), forbidden_words: frozenset = frozenset({'be'}), temperature: float = 1.0, pos: frozenset = frozenset({'VERB', 'ADJ', 'ADP', 'NOUN', 'ADV', 'DET'}), use_proba: bool = False, kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `present` | `tuple` |  | Word index in the text for the words in the proposed anchor. |
| `n` | `int` |  | Number of samples used when sampling from the corpus. |
| `sample_proba` | `float` | `0.5` | Sample probability for a word if `use_proba=False`. |
| `forbidden` | `frozenset` | `frozenset()` | Forbidden lemmas. |
| `forbidden_tags` | `frozenset` | `frozenset({'PRP$'})` | Forbidden POS tags. |
| `forbidden_words` | `frozenset` | `frozenset({'be'})` | Forbidden words. |
| `temperature` | `float` | `1.0` | Sample weight hyper-parameter if ``use_proba=True``. |
| `pos` | `frozenset` | `frozenset({'VERB', 'ADJ', 'ADP', 'NOUN', 'ADV', 'DET'})` | POS that can be changed during perturbation. |
| `use_proba` | `bool` | `False` | Bool whether to sample according to a similarity score with the corpus embeddings. |
| `**kwargs` |  |  | Other arguments. Not used. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  | Text to be processed. |

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
| `nlp` | `spacy.language.Language` |  | `spaCy` object. |
| `perturb_opts` | `Dict` |  | Perturbation options. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `text` | `str` |  | Text to be processed. |

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

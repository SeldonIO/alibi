# `alibi.explainers.anchors.language_model_text_sampler`
## `LanguageModelSampler`

_Inherits from:_ `AnchorTextSampler`

### Constructor

```python
LanguageModelSampler(self, model: alibi.utils.lang_model.LanguageModel, perturb_opts: dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `model` | `alibi.utils.lang_model.LanguageModel` |  |  |
| `perturb_opts` | `dict` |  |  |

### Methods

#### `create_mask`

```python
create_mask(anchor: tuple, num_samples: int, sample_proba: float = 1.0, filling: str = 'parallel', frac_mask_templates: float = 0.1, kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Create mask for words to be perturbed.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |
| `num_samples` | `int` |  |  |
| `sample_proba` | `float` | `1.0` |  |
| `filling` | `str` | `'parallel'` |  |
| `frac_mask_templates` | `float` | `0.1` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `fill_mask`

```python
fill_mask(raw: numpy.ndarray, data: numpy.ndarray, num_samples: int, top_n: int = 100, batch_size_lm: int = 32, filling: str = 'parallel', kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Fill in the masked tokens with language model.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `raw` | `numpy.ndarray` |  |  |
| `data` | `numpy.ndarray` |  |  |
| `num_samples` | `int` |  |  |
| `top_n` | `int` | `100` |  |
| `batch_size_lm` | `int` | `32` |  |
| `filling` | `str` | `'parallel'` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `get_sample_ids`

```python
get_sample_ids(punctuation: str = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', stopwords: Optional[List[str]] = None, kwargs) -> None
```

Find indices in words which can be perturbed.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `punctuation` | `str` | `'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'` |  |
| `stopwords` | `Optional[List[str]]` | `None` |  |

**Returns**
- Type: `None`

#### `perturb_sentence`

```python
perturb_sentence(anchor: tuple, num_samples: int, sample_proba: float = 0.5, top_n: int = 100, batch_size_lm: int = 32, filling: str = 'parallel', kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]
```

The function returns an `numpy` array of `num_samples` where randomly chosen features,

except those in anchor, are replaced by words sampled according to the language
model's predictions.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |
| `num_samples` | `int` |  |  |
| `sample_proba` | `float` | `0.5` |  |
| `top_n` | `int` | `100` |  |
| `batch_size_lm` | `int` | `32` |  |
| `filling` | `str` | `'parallel'` |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

#### `seed`

```python
seed(seed: int) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `seed` | `int` |  |  |

**Returns**
- Type: `None`

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

# `alibi.explainers.anchors.anchor_image`
## Constants
### `DEFAULT_DATA_ANCHOR_IMG`
```python
DEFAULT_DATA_ANCHOR_IMG: dict = {'anchor': [], 'segments': None, 'precision': None, 'coverage': None, 'raw': ...
```

### `DEFAULT_META_ANCHOR`
```python
DEFAULT_META_ANCHOR: dict = {'name': None, 'type': ['blackbox'], 'explanations': ['local'], 'params': {},...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.anchors.anchor_image (WARNING)>
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

### `DEFAULT_SEGMENTATION_KWARGS`
```python
DEFAULT_SEGMENTATION_KWARGS: dict = {'felzenszwalb': {}, 'quickshift': {}, 'slic': {'n_segments': 10, 'compactnes...
```

## `AnchorImage`

_Inherits from:_ `Explainer`, `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Constructor

```python
AnchorImage(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], image_shape: tuple, dtype: Type[numpy.generic] = <class 'numpy.float32'>, segmentation_fn: Any = 'slic', segmentation_kwargs: Optional[dict] = None, images_background: Optional[numpy.ndarray] = None, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `image_shape` | `tuple` |  |  |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` |  |
| `segmentation_fn` | `typing.Any` | `'slic'` |  |
| `segmentation_kwargs` | `Optional[dict]` | `None` |  |
| `images_background` | `Optional[numpy.ndarray]` | `None` |  |
| `seed` | `Optional[int]` | `None` |  |

### Methods

### `explain`

```python
explain(image: numpy.ndarray, p_sample: float = 0.5, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

Explain instance and return anchor with metadata.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |
| `p_sample` | `float` | `0.5` |  |
| `threshold` | `float` | `0.95` |  |
| `delta` | `float` | `0.1` |  |
| `tau` | `float` | `0.15` |  |
| `batch_size` | `int` | `100` |  |
| `coverage_samples` | `int` | `10000` |  |
| `beam_size` | `int` | `1` |  |
| `stop_on_first` | `bool` | `False` |  |
| `max_anchor_size` | `Optional[int]` | `None` |  |
| `min_samples_start` | `int` | `100` |  |
| `n_covered_ex` | `int` | `10` |  |
| `binary_cache_size` | `int` | `10000` |  |
| `cache_margin` | `int` | `1000` |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

Generates superpixels from (i.e., segments) an image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `overlay_mask`

```python
overlay_mask(image: numpy.ndarray, segments: numpy.ndarray, mask_features: list, scale: tuple = (0, 255)) -> numpy.ndarray
```

Overlay image with mask described by the mask features.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |
| `segments` | `numpy.ndarray` |  |  |
| `mask_features` | `list` |  |  |
| `scale` | `tuple` | `(0, 255)` |  |

**Returns**
- Type: `numpy.ndarray`

### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

## `AnchorImageSampler`

### Constructor

```python
AnchorImageSampler(self, predictor: Callable, segmentation_fn: Callable, custom_segmentation: bool, image: numpy.ndarray, images_background: Optional[numpy.ndarray] = None, p_sample: float = 0.5, n_covered_ex: int = 10)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |
| `segmentation_fn` | `Callable` |  |  |
| `custom_segmentation` | `bool` |  |  |
| `image` | `numpy.ndarray` |  |  |
| `images_background` | `Optional[numpy.ndarray]` | `None` |  |
| `p_sample` | `float` | `0.5` |  |
| `n_covered_ex` | `int` | `10` |  |

### Methods

### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

Compute the agreement between a classifier prediction on an instance to be explained

and the prediction on a set of samples which have a subset of perturbed superpixels.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

Generates superpixels from (i.e., segments) an image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Perturbs an image by altering the values of selected superpixels. If a dataset of image

backgrounds is provided to the explainer, then the superpixels are replaced with the
equivalent superpixels from the background image. Otherwise, the superpixels are replaced
by their average value.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  |  |
| `num_samples` | `int` |  |  |

**Returns**
- Type: `Tuple[numpy.ndarray, numpy.ndarray]`

## Functions
### `scale_image`

```python
scale_image(image: numpy.ndarray, scale: tuple = (0, 255)) -> numpy.ndarray
```

Scales an image in a specified range.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |
| `scale` | `tuple` | `(0, 255)` |  |

**Returns**
- Type: `numpy.ndarray`

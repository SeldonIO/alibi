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

### Constructor

```python
AnchorImage(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], image_shape: tuple, dtype: Type[numpy.generic] = <class 'numpy.float32'>, segmentation_fn: Any = 'slic', segmentation_kwargs: Optional[dict] = None, images_background: Optional[numpy.ndarray] = None, seed: Optional[int] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. |
| `image_shape` | `tuple` |  | Shape of the image to be explained. The channel axis is expected to be last. |
| `dtype` | `type[numpy.generic]` | `<class 'numpy.float32'>` | A `numpy` scalar type that corresponds to the type of input array expected by `predictor`. This may be used to construct arrays of the given type to be passed through the `predictor`. For most use cases this argument should have no effect, but it is exposed for use with predictors that would break when called with an array of unsupported type. |
| `segmentation_fn` | `typing.Any` | `'slic'` | Any of the built in segmentation function strings: ``'felzenszwalb'``, ``'slic'`` or ``'quickshift'`` or a custom segmentation function (callable) which returns an image mask with labels for each superpixel. The segmentation function is expected to return a segmentation mask containing all integer values from `0` to `K-1`, where `K` is the number of image segments (superpixels). See http://scikit-image.org/docs/dev/api/skimage.segmentation.html for more info. |
| `segmentation_kwargs` | `Optional[dict]` | `None` | Keyword arguments for the built in segmentation functions. |
| `images_background` | `Optional[numpy.ndarray]` | `None` | Images to overlay superpixels on. |
| `seed` | `Optional[int]` | `None` | If set, ensures different runs with the same input will yield same explanation. |

### Methods

#### `explain`

```python
explain(image: numpy.ndarray, p_sample: float = 0.5, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  | Image to be explained. |
| `p_sample` | `float` | `0.5` | The probability of simulating the absence of a superpixel. If the `images_background` is not provided, the absent superpixels will be replaced by the average value of their constituent pixels. Otherwise, the synthetic instances are created by fixing the present superpixels and superimposing another image from the `images_background` over the rest of the absent superpixels. |
| `threshold` | `float` | `0.95` | Minimum anchor precision threshold. The algorithm tries to find an anchor that maximizes the coverage under precision constraint. The precision constraint is formally defined as :math:`P(prec(A) \ge t) \ge 1 - \delta`, where :math:`A` is an anchor, :math:`t` is the `threshold` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, we are seeking for an anchor having its precision greater or equal than the given `threshold` with a confidence of `(1 - delta)`. A higher value guarantees that the anchors are faithful to the model, but also leads to more computation time. Note that there are cases in which the precision constraint cannot be satisfied due to the quantile-based discretisation of the numerical features. If that is the case, the best (i.e. highest coverage) non-eligible anchor is returned. |
| `delta` | `float` | `0.1` | Significance threshold. `1 - delta` represents the confidence threshold for the anchor precision (see `threshold`) and the selection of the best anchor candidate in each iteration (see `tau`). |
| `tau` | `float` | `0.15` | Multi-armed bandit parameter used to select candidate anchors in each iteration. The multi-armed bandit algorithm tries to find within a tolerance `tau` the most promising (i.e. according to the precision) `beam_size` candidate anchor(s) from a list of proposed anchors. Formally, when the `beam_size=1`, the multi-armed bandit algorithm seeks to find an anchor :math:`A` such that :math:`P(prec(A) \ge prec(A^\star) - \tau) \ge 1 - \delta`, where :math:`A^\star` is the anchor with the highest true precision (which we don't know), :math:`\tau` is the `tau` parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor. In other words, in each iteration, the algorithm returns with a probability of at least `1 - delta` an anchor :math:`A` with a precision within an error tolerance of `tau` from the precision of the highest true precision anchor :math:`A^\star`. A bigger value for `tau` means faster convergence but also looser anchor conditions. |
| `batch_size` | `int` | `100` | Batch size used for sampling. The Anchor algorithm will query the black-box model in batches of size `batch_size`. A larger `batch_size` gives more confidence in the anchor, again at the expense of computation time since it involves more model prediction calls. |
| `coverage_samples` | `int` | `10000` | Number of samples used to estimate coverage from during result search. |
| `beam_size` | `int` | `1` | Number of candidate anchors selected by the multi-armed bandit algorithm in each iteration from a list of proposed anchors. A bigger beam  width can lead to a better overall anchor (i.e. prevents the algorithm of getting stuck in a local maximum) at the expense of more computation time. |
| `stop_on_first` | `bool` | `False` | If ``True``, the beam search algorithm will return the first anchor that has satisfies the probability constraint. |
| `max_anchor_size` | `Optional[int]` | `None` | Maximum number of features in result. |
| `min_samples_start` | `int` | `100` | Min number of initial samples. |
| `n_covered_ex` | `int` | `10` | How many examples where anchors apply to store for each anchor sampled during search (both examples where prediction on samples agrees/disagrees with `desired_label` are stored). |
| `binary_cache_size` | `int` | `10000` | The result search pre-allocates `binary_cache_size` batches for storing the binary arrays returned during sampling. |
| `cache_margin` | `int` | `1000` | When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache of the same size is pre-allocated to continue buffering samples. |
| `verbose` | `bool` | `False` | Display updates during the anchor search iterations. |
| `verbose_every` | `int` | `1` | Frequency of displayed iterations during anchor search process. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  | A grayscale or RGB image. |

**Returns**
- Type: `numpy.ndarray`

#### `overlay_mask`

```python
overlay_mask(image: numpy.ndarray, segments: numpy.ndarray, mask_features: list, scale: tuple = (0, 255)) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  | Image to be explained. |
| `segments` | `numpy.ndarray` |  | Superpixels. |
| `mask_features` | `list` |  | List with superpixels present in mask. |
| `scale` | `tuple` | `(0, 255)` | Pixel scale for masked image. |

**Returns**
- Type: `numpy.ndarray`

#### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | New predictor function. |

**Returns**
- Type: `None`

## `AnchorImageSampler`

### Constructor

```python
AnchorImageSampler(self, predictor: Callable, segmentation_fn: Callable, custom_segmentation: bool, image: numpy.ndarray, images_background: Optional[numpy.ndarray] = None, p_sample: float = 0.5, n_covered_ex: int = 10)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. |
| `segmentation_fn` | `Callable` |  | Function used to segment the images. The segmentation function is expected to return a segmentation mask containing all integer values from `0` to `K-1`, where `K` is the number of image segments (superpixels). |
| `custom_segmentation` | `bool` |  |  |
| `image` | `numpy.ndarray` |  | Image to be explained. |
| `images_background` | `Optional[numpy.ndarray]` | `None` | Images to overlay superpixels on. |
| `p_sample` | `float` | `0.5` | Probability for a pixel to be represented by the average value of its superpixel. |
| `n_covered_ex` | `int` | `10` | How many examples where anchors apply to store for each anchor sampled during search (both examples where prediction on samples agrees/disagrees with `desired_label` are stored). |

### Methods

#### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  | Samples whose labels are to be compared with the instance label. |

**Returns**
- Type: `numpy.ndarray`

#### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  | A grayscale or RGB image. |

**Returns**
- Type: `numpy.ndarray`

#### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchor` | `tuple` |  | Contains the superpixels whose values are not going to be perturbed. |
| `num_samples` | `int` |  | Number of perturbed samples to be returned. |

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
| `image` | `numpy.ndarray` |  | Image to be scale. |
| `scale` | `tuple` | `(0, 255)` | The scaling interval. |

**Returns**
- Type: `numpy.ndarray`

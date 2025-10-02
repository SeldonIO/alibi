# `alibi.explainers.anchors.anchor_image`
## Classes
### `AnchorImage` (_inherits from `Explainer`, `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

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

#### Methods

##### `explain`

```python
explain(image: numpy.ndarray, p_sample: float = 0.5, threshold: float = 0.95, delta: float = 0.1, tau: float = 0.15, batch_size: int = 100, coverage_samples: int = 10000, beam_size: int = 1, stop_on_first: bool = False, max_anchor_size: Optional[int] = None, min_samples_start: int = 100, n_covered_ex: int = 10, binary_cache_size: int = 10000, cache_margin: int = 1000, verbose: bool = False, verbose_every: int = 1, kwargs: typing.Any) -> alibi.api.interfaces.Explanation
```

Explain instance and return anchor with metadata.

Parameters
----------
image
    Image to be explained.
p_sample
    The probability of simulating the absence of a superpixel. If the `images_background` is not provided,
    the absent superpixels will be replaced by the average value of their constituent pixels. Otherwise,
    the synthetic instances are created by fixing the present superpixels and superimposing another image
    from the `images_background` over the rest of the absent superpixels.
threshold
    Minimum anchor precision threshold. The algorithm tries to find an anchor that maximizes the coverage
    under precision constraint. The precision constraint is formally defined as
    :math:`P(prec(A) \ge t) \ge 1 - \delta`, where :math:`A` is an anchor, :math:`t` is the `threshold`
    parameter, :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision
    of an anchor. In other words, we are seeking for an anchor having its precision greater or equal than
    the given `threshold` with a confidence of `(1 - delta)`. A higher value guarantees that the anchors are
    faithful to the model, but also leads to more computation time. Note that there are cases in which the
    precision constraint cannot be satisfied due to the quantile-based discretisation of the numerical
    features. If that is the case, the best (i.e. highest coverage) non-eligible anchor is returned.
delta
    Significance threshold. `1 - delta` represents the confidence threshold for the anchor precision
    (see `threshold`) and the selection of the best anchor candidate in each iteration (see `tau`).
tau
    Multi-armed bandit parameter used to select candidate anchors in each iteration. The multi-armed bandit
    algorithm tries to find within a tolerance `tau` the most promising (i.e. according to the precision)
    `beam_size` candidate anchor(s) from a list of proposed anchors. Formally, when the `beam_size=1`,
    the multi-armed bandit algorithm seeks to find an anchor :math:`A` such that
    :math:`P(prec(A) \ge prec(A^\star) - \tau) \ge 1 - \delta`, where :math:`A^\star` is the anchor
    with the highest true precision (which we don't know), :math:`\tau` is the `tau` parameter,
    :math:`\delta` is the `delta` parameter, and :math:`prec(\cdot)` denotes the precision of an anchor.
    In other words, in each iteration, the algorithm returns with a probability of at least `1 - delta` an
    anchor :math:`A` with a precision within an error tolerance of `tau` from the precision of the
    highest true precision anchor :math:`A^\star`. A bigger value for `tau` means faster convergence but also
    looser anchor conditions.
batch_size
    Batch size used for sampling. The Anchor algorithm will query the black-box model in batches of size
    `batch_size`. A larger `batch_size` gives more confidence in the anchor, again at the expense of
    computation time since it involves more model prediction calls.
coverage_samples
    Number of samples used to estimate coverage from during result search.
beam_size
    Number of candidate anchors selected by the multi-armed bandit algorithm in each iteration from a list of
    proposed anchors. A bigger beam  width can lead to a better overall anchor (i.e. prevents the algorithm
    of getting stuck in a local maximum) at the expense of more computation time.
stop_on_first
    If ``True``, the beam search algorithm will return the first anchor that has satisfies the
    probability constraint.
max_anchor_size
    Maximum number of features in result.
min_samples_start
    Min number of initial samples.
n_covered_ex
    How many examples where anchors apply to store for each anchor sampled during search
    (both examples where prediction on samples agrees/disagrees with `desired_label` are stored).
binary_cache_size
    The result search pre-allocates `binary_cache_size` batches for storing the binary arrays
    returned during sampling.
cache_margin
    When only ``max(cache_margin, batch_size)`` positions in the binary cache remain empty, a new cache
    of the same size is pre-allocated to continue buffering samples.
verbose
    Display updates during the anchor search iterations.
verbose_every
    Frequency of displayed iterations during anchor search process.

Returns
-------
explanation
    `Explanation` object containing the anchor explaining the instance with additional metadata as attributes.
    See usage at `AnchorImage examples`_ for details.

    .. _AnchorImage examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html

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
| `kwargs` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

Generates superpixels from (i.e., segments) an image.

Parameters
----------
image
    A grayscale or RGB image.

Returns
-------
A `[H, W]` array of integers. Each integer is a segment (superpixel) label.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

##### `overlay_mask`

```python
overlay_mask(image: numpy.ndarray, segments: numpy.ndarray, mask_features: list, scale: tuple = (0, 255)) -> numpy.ndarray
```

Overlay image with mask described by the mask features.

Parameters
----------
image
    Image to be explained.
segments
    Superpixels.
mask_features
    List with superpixels present in mask.
scale
    Pixel scale for masked image.

Returns
-------
masked_image
    Image overlaid with mask.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |
| `segments` | `numpy.ndarray` |  |  |
| `mask_features` | `list` |  |  |
| `scale` | `tuple` | `(0, 255)` |  |

**Returns**
- Type: `numpy.ndarray`

##### `reset_predictor`

```python
reset_predictor(predictor: Callable) -> None
```

Resets the predictor function.

Parameters
----------
predictor
    New predictor function.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable` |  |  |

**Returns**
- Type: `None`

### `AnchorImageSampler`

#### Constructor

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

#### Methods

##### `compare_labels`

```python
compare_labels(samples: numpy.ndarray) -> numpy.ndarray
```

Compute the agreement between a classifier prediction on an instance to be explained

and the prediction on a set of samples which have a subset of perturbed superpixels.

Parameters
----------
samples
    Samples whose labels are to be compared with the instance label.

Returns
-------
A boolean array indicating whether the prediction was the same as the instance label.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samples` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

##### `generate_superpixels`

```python
generate_superpixels(image: numpy.ndarray) -> numpy.ndarray
```

Generates superpixels from (i.e., segments) an image.

Parameters
----------
image
    A grayscale or RGB image.

Returns
-------
 A `[H, W]` array of integers. Each integer is a segment (superpixel) label.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

##### `perturbation`

```python
perturbation(anchor: tuple, num_samples: int) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Perturbs an image by altering the values of selected superpixels. If a dataset of image

backgrounds is provided to the explainer, then the superpixels are replaced with the
equivalent superpixels from the background image. Otherwise, the superpixels are replaced
by their average value.

Parameters
----------
anchor:
    Contains the superpixels whose values are not going to be perturbed.
num_samples:
    Number of perturbed samples to be returned.

Returns
-------
imgs
    A `[num_samples, H, W, C]` array of perturbed images.
segments_mask
    A `[num_samples, M]` binary mask, where `M` is the number of image superpixels
    segments. 1 indicates the values in that particular superpixels are not
    perturbed.

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

Parameters
----------
image
    Image to be scale.
scale
    The scaling interval.

Returns
-------
img_scaled
    Scaled image.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `image` | `numpy.ndarray` |  |  |
| `scale` | `tuple` | `(0, 255)` |  |

**Returns**
- Type: `numpy.ndarray`

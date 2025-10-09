# `alibi.explainers.anchors.anchor_base`
## Constants
### `logger`
```python
logger: Logger = <Logger alibi.explainers.anchors.anchor_base (WARNING)>
```
## `AnchorBaseBeam`

### Constructor

```python
AnchorBaseBeam(self, samplers: List[Callable], **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samplers` | `List[Callable]` |  |  |

### Methods

#### `anchor_beam`

```python
anchor_beam(delta: float = 0.05, epsilon: float = 0.1, desired_confidence: float = 1.0, beam_size: int = 1, epsilon_stop: float = 0.05, min_samples_start: int = 100, max_anchor_size: Optional[int] = None, stop_on_first: bool = False, batch_size: int = 100, coverage_samples: int = 10000, verbose: bool = False, verbose_every: int = 1, kwargs) -> dict
```

Uses the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013) together with additional sampling to search

feature sets (anchors) that guarantee the prediction made by a classifier model. The search is greedy if
``beam_size=1``. Otherwise, at each of the `max_anchor_size` steps, `beam_size` solutions are explored.
By construction, solutions found have high precision (defined as the expected of number of times the classifier
makes the same prediction when queried with the feature subset combined with arbitrary samples drawn from a
noise distribution). The algorithm maximises the coverage of the solution found - the frequency of occurrence
of records containing the feature subset in set of samples.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `delta` | `float` | `0.05` |  |
| `epsilon` | `float` | `0.1` |  |
| `desired_confidence` | `float` | `1.0` |  |
| `beam_size` | `int` | `1` |  |
| `epsilon_stop` | `float` | `0.05` |  |
| `min_samples_start` | `int` | `100` |  |
| `max_anchor_size` | `Optional[int]` | `None` |  |
| `stop_on_first` | `bool` | `False` |  |
| `batch_size` | `int` | `100` |  |
| `coverage_samples` | `int` | `10000` |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |

**Returns**
- Type: `dict`

#### `compute_beta`

```python
compute_beta(n_features: int, t: int, delta: float) -> float
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_features` | `int` |  |  |
| `t` | `int` |  |  |
| `delta` | `float` |  |  |

**Returns**
- Type: `float`

#### `dlow_bernoulli`

```python
dlow_bernoulli(p: numpy.ndarray, level: numpy.ndarray, n_iter: int = 17) -> numpy.ndarray
```

Update lower precision bound for a candidate anchors dependent on the KL-divergence.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  |  |
| `level` | `numpy.ndarray` |  |  |
| `n_iter` | `int` | `17` |  |

**Returns**
- Type: `numpy.ndarray`

#### `draw_samples`

```python
draw_samples(anchors: list, batch_size: int) -> Tuple[tuple, tuple]
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  |  |
| `batch_size` | `int` |  |  |

**Returns**
- Type: `Tuple[tuple, tuple]`

#### `dup_bernoulli`

```python
dup_bernoulli(p: numpy.ndarray, level: numpy.ndarray, n_iter: int = 17) -> numpy.ndarray
```

Update upper precision bound for a candidate anchors dependent on the KL-divergence.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  |  |
| `level` | `numpy.ndarray` |  |  |
| `n_iter` | `int` | `17` |  |

**Returns**
- Type: `numpy.ndarray`

#### `get_anchor_metadata`

```python
get_anchor_metadata(features: tuple, success, batch_size: int = 100) -> dict
```

Given the features contained in a result, it retrieves metadata such as the precision and

coverage of the result and partial anchors and examples where the result/partial anchors
apply and yield the same prediction as on the instance to be explained (`covered_true`)
or a different prediction (`covered_false`).

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `features` | `tuple` |  |  |
| `success` |  |  |  |
| `batch_size` | `int` | `100` |  |

**Returns**
- Type: `dict`

#### `get_init_stats`

```python
get_init_stats(anchors: list, coverages = False) -> dict
```

Finds the number of samples already drawn for each result in anchors, their

comparisons with the instance to be explained and, optionally, coverage.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  |  |
| `coverages` |  | `False` |  |

**Returns**
- Type: `dict`

#### `kllucb`

```python
kllucb(anchors: list, init_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int, verbose: bool = False, verbose_every: int = 1) -> numpy.ndarray
```

Implements the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013).

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  |  |
| `init_stats` | `dict` |  |  |
| `epsilon` | `float` |  |  |
| `delta` | `float` |  |  |
| `batch_size` | `int` |  |  |
| `top_n` | `int` |  |  |
| `verbose` | `bool` | `False` |  |
| `verbose_every` | `int` | `1` |  |

**Returns**
- Type: `numpy.ndarray`

#### `propose_anchors`

```python
propose_anchors(previous_best: list) -> list
```

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `previous_best` | `list` |  |  |

**Returns**
- Type: `list`

#### `select_critical_arms`

```python
select_critical_arms(means: numpy.ndarray, ub: numpy.ndarray, lb: numpy.ndarray, n_samples: numpy.ndarray, delta: float, top_n: int, t: int)
```

Determines a set of two anchors by updating the upper bound for low empirical precision anchors and

the lower bound for anchors with high empirical precision.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `means` | `numpy.ndarray` |  |  |
| `ub` | `numpy.ndarray` |  |  |
| `lb` | `numpy.ndarray` |  |  |
| `n_samples` | `numpy.ndarray` |  |  |
| `delta` | `float` |  |  |
| `top_n` | `int` |  |  |
| `t` | `int` |  |  |

#### `to_sample`

```python
to_sample(means: numpy.ndarray, ubs: numpy.ndarray, lbs: numpy.ndarray, desired_confidence: float, epsilon_stop: float)
```

Given an array of mean result precisions and their upper and lower bounds, determines for which anchors

more samples need to be drawn in order to estimate the anchors precision with `desired_confidence` and error
tolerance.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `means` | `numpy.ndarray` |  |  |
| `ubs` | `numpy.ndarray` |  |  |
| `lbs` | `numpy.ndarray` |  |  |
| `desired_confidence` | `float` |  |  |
| `epsilon_stop` | `float` |  |  |

#### `update_state`

```python
update_state(covered_true: numpy.ndarray, covered_false: numpy.ndarray, labels: numpy.ndarray, samples: Tuple[numpy.ndarray, float], anchor: tuple) -> Tuple[int, int]
```

Updates the explainer state (see :py:meth:`alibi.explainers.anchors.anchor_base.AnchorBaseBeam.__init__`

for full state definition).

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `covered_true` | `numpy.ndarray` |  |  |
| `covered_false` | `numpy.ndarray` |  |  |
| `labels` | `numpy.ndarray` |  |  |
| `samples` | `Tuple[numpy.ndarray, float]` |  |  |
| `anchor` | `tuple` |  |  |

**Returns**
- Type: `Tuple[int, int]`

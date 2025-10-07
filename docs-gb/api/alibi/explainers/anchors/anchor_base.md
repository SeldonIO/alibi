# `alibi.explainers.anchors.anchor_base`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.anchors.anchor_base (WARNING)>
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

## `AnchorBaseBeam`

### Constructor

```python
AnchorBaseBeam(self, samplers: List[Callable], **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `samplers` | `List[Callable]` |  | Objects that can be called with args (`result`, `n_samples`) tuple to draw samples. |

### Methods

#### `anchor_beam`

```python
anchor_beam(delta: float = 0.05, epsilon: float = 0.1, desired_confidence: float = 1.0, beam_size: int = 1, epsilon_stop: float = 0.05, min_samples_start: int = 100, max_anchor_size: Optional[int] = None, stop_on_first: bool = False, batch_size: int = 100, coverage_samples: int = 10000, verbose: bool = False, verbose_every: int = 1, kwargs) -> dict
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `delta` | `float` | `0.05` | Used to compute `beta`. |
| `epsilon` | `float` | `0.1` | Precision bound tolerance for convergence. |
| `desired_confidence` | `float` | `1.0` | Desired level of precision (`tau` in `paper <https://homes.cs.washington.edu/~marcotcr/aaai18.pdf>`_). |
| `beam_size` | `int` | `1` | Beam width. |
| `epsilon_stop` | `float` | `0.05` | Confidence bound margin around desired precision. |
| `min_samples_start` | `int` | `100` | Min number of initial samples. |
| `max_anchor_size` | `Optional[int]` | `None` | Max number of features in result. |
| `stop_on_first` | `bool` | `False` | Stop on first valid result found. |
| `batch_size` | `int` | `100` | Number of samples used for an arm evaluation. |
| `coverage_samples` | `int` | `10000` | Number of samples from which to build a coverage set. |
| `verbose` | `bool` | `False` | Whether to print intermediate LUCB & anchor selection output. |
| `verbose_every` | `int` | `1` | Print intermediate output every verbose_every steps. |

**Returns**
- Type: `dict`

#### `compute_beta`

```python
compute_beta(n_features: int, t: int, delta: float) -> float
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `n_features` | `int` |  | Number of candidate anchors. |
| `t` | `int` |  | Iteration number. |
| `delta` | `float` |  | Confidence budget, candidate anchors have close to optimal precisions with prob. `1 - delta`. |

**Returns**
- Type: `float`

#### `dlow_bernoulli`

```python
dlow_bernoulli(p: numpy.ndarray, level: numpy.ndarray, n_iter: int = 17) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  | Precision of candidate anchors. |
| `level` | `numpy.ndarray` |  | `beta / nb of samples` for each result. |
| `n_iter` | `int` | `17` | Number of iterations during lower bound update. |

**Returns**
- Type: `numpy.ndarray`

#### `draw_samples`

```python
draw_samples(anchors: list, batch_size: int) -> Tuple[tuple, tuple]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  | Anchors on which samples are conditioned. |
| `batch_size` | `int` |  | The number of samples drawn for each result. |

**Returns**
- Type: `Tuple[tuple, tuple]`

#### `dup_bernoulli`

```python
dup_bernoulli(p: numpy.ndarray, level: numpy.ndarray, n_iter: int = 17) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  | Precision of candidate anchors. |
| `level` | `numpy.ndarray` |  | `beta / nb of samples` for each result. |
| `n_iter` | `int` | `17` | Number of iterations during lower bound update. |

**Returns**
- Type: `numpy.ndarray`

#### `get_anchor_metadata`

```python
get_anchor_metadata(features: tuple, success, batch_size: int = 100) -> dict
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `features` | `tuple` |  | Sorted indices of features in result. |
| `success` |  |  | Indicates whether an anchor satisfying precision threshold was met or not. |
| `batch_size` | `int` | `100` | Number of samples among which positive and negative examples for partial anchors are selected if partial anchors have not already been explicitly sampled. |

**Returns**
- Type: `dict`

#### `get_init_stats`

```python
get_init_stats(anchors: list, coverages = False) -> dict
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  | Candidate anchors. |
| `coverages` |  | `False` | If ``True``, the statistics returned contain the coverage of the specified anchors. |

**Returns**
- Type: `dict`

#### `kllucb`

```python
kllucb(anchors: list, init_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int, verbose: bool = False, verbose_every: int = 1) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `anchors` | `list` |  | A list of anchors from which two critical anchors are selected (see Kaufmann and Kalyanakrishnan, 2013). |
| `init_stats` | `dict` |  | Dictionary with lists containing nb of samples used and where sample predictions equal the desired label. |
| `epsilon` | `float` |  | Precision bound tolerance for convergence. |
| `delta` | `float` |  | Used to compute `beta`. |
| `batch_size` | `int` |  | Number of samples. |
| `top_n` | `int` |  | Min of beam width size or number of candidate anchors. |
| `verbose` | `bool` | `False` | Whether to print intermediate output. |
| `verbose_every` | `int` | `1` | Whether to print intermediate output every `verbose_every` steps. |

**Returns**
- Type: `numpy.ndarray`

#### `propose_anchors`

```python
propose_anchors(previous_best: list) -> list
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `previous_best` | `list` |  | List with tuples of result candidates. |

**Returns**
- Type: `list`

#### `select_critical_arms`

```python
select_critical_arms(means: numpy.ndarray, ub: numpy.ndarray, lb: numpy.ndarray, n_samples: numpy.ndarray, delta: float, top_n: int, t: int)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `means` | `numpy.ndarray` |  | Empirical mean result precisions. |
| `ub` | `numpy.ndarray` |  | Upper bound on result precisions. |
| `lb` | `numpy.ndarray` |  | Lower bound on result precisions. |
| `n_samples` | `numpy.ndarray` |  | The number of samples drawn for each candidate result. |
| `delta` | `float` |  | Confidence budget, candidate anchors have close to optimal precisions with prob. `1 - delta`. |
| `top_n` | `int` |  | Number of arms to be selected. |
| `t` | `int` |  | Iteration number. |

#### `to_sample`

```python
to_sample(means: numpy.ndarray, ubs: numpy.ndarray, lbs: numpy.ndarray, desired_confidence: float, epsilon_stop: float)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `means` | `numpy.ndarray` |  | Mean precisions (each element represents a different result). |
| `ubs` | `numpy.ndarray` |  | Precisions' upper bounds (each element represents a different result). |
| `lbs` | `numpy.ndarray` |  | Precisions' lower bounds (each element represents a different result). |
| `desired_confidence` | `float` |  | Desired level of confidence for precision estimation. |
| `epsilon_stop` | `float` |  | Tolerance around desired precision. |

#### `update_state`

```python
update_state(covered_true: numpy.ndarray, covered_false: numpy.ndarray, labels: numpy.ndarray, samples: Tuple[numpy.ndarray, float], anchor: tuple) -> Tuple[int, int]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `covered_true` | `numpy.ndarray` |  | Examples where the result applies and the prediction is the same as on the instance to be explained. |
| `covered_false` | `numpy.ndarray` |  | Examples where the result applies and the prediction is the different to the instance to be explained. |
| `labels` | `numpy.ndarray` |  | An array indicating whether the prediction on the sample matches the label of the instance to be explained. |
| `samples` | `Tuple[numpy.ndarray, float]` |  | A tuple containing discretized data, coverage and the result sampled. |
| `anchor` | `tuple` |  | The result to be updated. |

**Returns**
- Type: `Tuple[int, int]`

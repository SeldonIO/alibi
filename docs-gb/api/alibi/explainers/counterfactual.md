# `alibi.explainers.counterfactual`
## `Counterfactual`

_Inherits from:_ `Explainer`, `ABC`, `Base`

### Constructor

```python
Counterfactual(self, predict_fn: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], shape: Tuple[int, ...], distance_fn: str = 'l1', target_proba: float = 1.0, target_class: Union[str, int] = 'other', max_iter: int = 1000, early_stop: int = 50, lam_init: float = 0.1, max_lam_steps: int = 10, tol: float = 0.05, learning_rate_init=0.1, feature_range: Union[Tuple, str] = (-10000000000.0, 10000000000.0), eps: Union[float, numpy.ndarray] = 0.01, init: str = 'identity', decay: bool = True, write_dir: Optional[str] = None, debug: bool = False, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict_fn` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  |  |
| `shape` | `Tuple[int, .Ellipsis]` |  |  |
| `distance_fn` | `str` | `'l1'` |  |
| `target_proba` | `float` | `1.0` |  |
| `target_class` | `Union[str, int]` | `'other'` |  |
| `max_iter` | `int` | `1000` |  |
| `early_stop` | `int` | `50` |  |
| `lam_init` | `float` | `0.1` |  |
| `max_lam_steps` | `int` | `10` |  |
| `tol` | `float` | `0.05` |  |
| `learning_rate_init` |  | `0.1` |  |
| `feature_range` | `Union[Tuple, str]` | `(-10000000000.0, 10000000000.0)` |  |
| `eps` | `Union[float, numpy.ndarray]` | `0.01` |  |
| `init` | `str` | `'identity'` |  |
| `decay` | `bool` | `True` |  |
| `write_dir` | `Optional[str]` | `None` |  |
| `debug` | `bool` | `False` |  |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray) -> alibi.api.interfaces.Explanation
```

Explain an instance and return the counterfactual with metadata.

Parameters
----------
X
    Instance to be explained.

Returns
-------
explanation
    `Explanation` object containing the counterfactual with additional metadata as attributes.
    See usage at `Counterfactual examples`_ for details.

    .. _Counterfactual examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/CF.html

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(X: numpy.ndarray, y: Optional[numpy.ndarray]) -> alibi.explainers.counterfactual.Counterfactual
```

Fit method - currently unused as the counterfactual search is fully unsupervised.

Parameters
----------
X
    Not used. Included for consistency.
y
    Not used. Included for consistency.

Returns
-------
self
    Explainer itself.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `y` | `Optional[numpy.ndarray]` |  |  |

**Returns**
- Type: `alibi.explainers.counterfactual.Counterfactual`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

Resets the predictor function/model.

Parameters
----------
predictor
    New predictor function/model.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  |  |

**Returns**
- Type: `None`

## Functions
### `CounterFactual`

```python
CounterFactual(args, kwargs)
```

The class name `CounterFactual` is deprecated, please use `Counterfactual`.

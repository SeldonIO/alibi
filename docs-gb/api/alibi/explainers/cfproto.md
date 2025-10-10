# `alibi.explainers.cfproto`
## Constants
### `DEFAULT_DATA_CFP`
```python
DEFAULT_DATA_CFP: dict = {'cf': None, 'all': [], 'orig_class': None, 'orig_proba': None, 'id_proto': N...
```

### `DEFAULT_META_CFP`
```python
DEFAULT_META_CFP: dict = {'name': None, 'type': ['blackbox', 'tensorflow', 'keras'], 'explanations': [...
```

### `logger`
```python
logger: logging.Logger = <Logger alibi.explainers.cfproto (WARNING)>
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

## `CounterfactualProto`

_Inherits from:_ `Explainer`, `FitMixin`, `ABC`, `Base`

### Constructor

```python
CounterfactualProto(self, predict: Union[Callable[[numpy.ndarray], numpy.ndarray], keras.src.models.model.Model], shape: tuple, kappa: float = 0.0, beta: float = 0.1, feature_range: Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]] = (-10000000000.0, 10000000000.0), gamma: float = 0.0, ae_model: Optional[keras.src.models.model.Model] = None, enc_model: Optional[keras.src.models.model.Model] = None, theta: float = 0.0, cat_vars: Optional[Dict[int, int]] = None, ohe: bool = False, use_kdtree: bool = False, learning_rate_init: float = 0.01, max_iterations: int = 1000, c_init: float = 10.0, c_steps: int = 10, eps: tuple = (0.001, 0.001), clip: tuple = (-1000.0, 1000.0), update_num_grad: int = 1, write_dir: Optional[str] = None, sess: Optional[tensorflow.python.client.session.Session] = None) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predict` | `Union[Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray], keras.src.models.model.Model]` |  | `tensorflow` model or any other model's prediction function returning class probabilities. |
| `shape` | `tuple` |  | Shape of input data starting with batch size. |
| `kappa` | `float` | `0.0` | Confidence parameter for the attack loss term. |
| `beta` | `float` | `0.1` | Regularization constant for L1 loss term. |
| `feature_range` | `Tuple[Union[float, numpy.ndarray], Union[float, numpy.ndarray]]` | `(-10000000000.0, 10000000000.0)` | Tuple with `min` and `max` ranges to allow for perturbed instances. `Min` and `max` ranges can be `float` or `numpy` arrays with dimension (1x nb of features) for feature-wise ranges. |
| `gamma` | `float` | `0.0` | Regularization constant for optional auto-encoder loss term. |
| `ae_model` | `Optional[keras.src.models.model.Model]` | `None` | Optional auto-encoder model used for loss regularization. |
| `enc_model` | `Optional[keras.src.models.model.Model]` | `None` | Optional encoder model used to guide instance perturbations towards a class prototype. |
| `theta` | `float` | `0.0` | Constant for the prototype search loss term. |
| `cat_vars` | `Optional[Dict[int, int]]` | `None` | Dict with as keys the categorical columns and as values the number of categories per categorical variable. |
| `ohe` | `bool` | `False` | Whether the categorical variables are one-hot encoded (OHE) or not. If not OHE, they are assumed to have ordinal encodings. |
| `use_kdtree` | `bool` | `False` | Whether to use k-d trees for the prototype loss term if no encoder is available. |
| `learning_rate_init` | `float` | `0.01` | Initial learning rate of optimizer. |
| `max_iterations` | `int` | `1000` | Maximum number of iterations for finding a counterfactual. |
| `c_init` | `float` | `10.0` | Initial value to scale the attack loss term. |
| `c_steps` | `int` | `10` | Number of iterations to adjust the constant scaling the attack loss term. |
| `eps` | `tuple` | `(0.001, 0.001)` | If numerical gradients are used to compute `dL/dx = (dL/dp) * (dp/dx)`, then `eps[0]` is used to calculate `dL/dp` and `eps[1]` is used for `dp/dx`. `eps[0]` and `eps[1]` can be a combination of `float` values and `numpy` arrays. For `eps[0]`, the array dimension should be (1x nb of prediction categories) and for `eps[1]` it should be (1x nb of features). |
| `clip` | `tuple` | `(-1000.0, 1000.0)` | Tuple with min and max clip ranges for both the numerical gradients and the gradients obtained from the `tensorflow` graph. |
| `update_num_grad` | `int` | `1` | If numerical gradients are used, they will be updated every `update_num_grad` iterations. |
| `write_dir` | `Optional[str]` | `None` | Directory to write `tensorboard` files to. |
| `sess` | `Optional[tensorflow.python.client.session.Session]` | `None` | Optional `tensorflow` session that will be used if passed instead of creating or inferring one internally. |

### Methods

#### `attack`

```python
attack(X: numpy.ndarray, Y: numpy.ndarray, target_class: Optional[list] = None, k: Optional[int] = None, k_type: str = 'mean', threshold: float = 0.0, verbose: bool = False, print_every: int = 100, log_every: int = 100) -> Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to attack. |
| `Y` | `numpy.ndarray` |  | Labels for `X` as one-hot-encoding. |
| `target_class` | `Optional[list]` | `None` | List with target classes used to find closest prototype. If ``None``, the nearest prototype except for the predict class on the instance is used. |
| `k` | `Optional[int]` | `None` | Number of nearest instances used to define the prototype for a class. Defaults to using all instances belonging to the class if an encoder is used and to 1 for k-d trees. |
| `k_type` | `str` | `'mean'` | Use either the average encoding of the k nearest instances in a class (``k_type='mean'``) or the k-nearest encoding in the class (``k_type='point'``) to define the prototype of that class. Only relevant if an encoder is used to define the prototypes. |
| `threshold` | `float` | `0.0` | Threshold level for the ratio between the distance of the counterfactual to the prototype of the predicted class for the original instance over the distance to the prototype of the predicted class for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does not meet the requirements. |
| `verbose` | `bool` | `False` | Print intermediate results of optimization if ``True``. |
| `print_every` | `int` | `100` | Print frequency if verbose is ``True``. |
| `log_every` | `int` | `100` | `tensorboard` log frequency if write directory is specified. |

**Returns**
- Type: `Tuple[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`

#### `explain`

```python
explain(X: numpy.ndarray, Y: Optional[numpy.ndarray] = None, target_class: Optional[list] = None, k: Optional[int] = None, k_type: str = 'mean', threshold: float = 0.0, verbose: bool = False, print_every: int = 100, log_every: int = 100) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instances to attack. |
| `Y` | `Optional[numpy.ndarray]` | `None` | Labels for `X` as one-hot-encoding. |
| `target_class` | `Optional[list]` | `None` | List with target classes used to find closest prototype. If ``None``, the nearest prototype except for the predict class on the instance is used. |
| `k` | `Optional[int]` | `None` | Number of nearest instances used to define the prototype for a class. Defaults to using all instances belonging to the class if an encoder is used and to 1 for k-d trees. |
| `k_type` | `str` | `'mean'` | Use either the average encoding of the `k` nearest instances in a class (``k_type='mean'``) or the k-nearest encoding in the class (``k_type='point'``) to define the prototype of that class. Only relevant if an encoder is used to define the prototypes. |
| `threshold` | `float` | `0.0` | Threshold level for the ratio between the distance of the counterfactual to the prototype of the predicted class for the original instance over the distance to the prototype of the predicted class for the counterfactual. If the trust score is below the threshold, the proposed counterfactual does not meet the requirements. |
| `verbose` | `bool` | `False` | Print intermediate results of optimization if ``True``. |
| `print_every` | `int` | `100` | Print frequency if verbose is ``True``. |
| `log_every` | `int` | `100` | `tensorboard` log frequency if write directory is specified |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(train_data: numpy.ndarray, trustscore_kwargs: Optional[dict] = None, d_type: str = 'abdm', w: Optional[float] = None, disc_perc: Sequence[Union[int, float]] = (25, 50, 75), standardize_cat_vars: bool = False, smooth: float = 1.0, center: bool = True, update_feature_range: bool = True) -> alibi.explainers.cfproto.CounterfactualProto
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `train_data` | `numpy.ndarray` |  | Representative sample from the training data. |
| `trustscore_kwargs` | `Optional[dict]` | `None` | Optional arguments to initialize the trust scores method. |
| `d_type` | `str` | `'abdm'` | Pairwise distance metric used for categorical variables. Currently, ``'abdm'``, ``'mvdm'`` and ``'abdm-mvdm'`` are supported. ``'abdm'`` infers context from the other variables while ``'mvdm'`` uses the model predictions. ``'abdm-mvdm'`` is a weighted combination of the two metrics. |
| `w` | `Optional[float]` | `None` | Weight on ``'abdm'`` (between 0. and 1.) distance if `d_type` equals ``'abdm-mvdm'``. |
| `disc_perc` | `Sequence[Union[int, float]]` | `(25, 50, 75)` | List with percentiles used in binning of numerical features used for the ``'abdm'`` and ``'abdm-mvdm'`` pairwise distance measures. |
| `standardize_cat_vars` | `bool` | `False` | Standardize numerical values of categorical variables if ``True``. |
| `smooth` | `float` | `1.0` | Smoothing exponent between 0 and 1 for the distances. Lower values will smooth the difference in distance metric between different features. |
| `center` | `bool` | `True` | Whether to center the scaled distance measures. If ``False``, the min distance for each feature except for the feature with the highest raw max distance will be the lower bound of the feature range, but the upper bound will be below the max feature range. |
| `update_feature_range` | `bool` | `True` | Update feature range with scaled values. |

**Returns**
- Type: `alibi.explainers.cfproto.CounterfactualProto`

#### `get_gradients`

```python
get_gradients(X: numpy.ndarray, Y: numpy.ndarray, grads_shape: tuple, cat_vars_ord: dict) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance around which gradient is evaluated. |
| `Y` | `numpy.ndarray` |  | One-hot representation of instance labels. |
| `grads_shape` | `tuple` |  | Shape of gradients. |
| `cat_vars_ord` | `dict` |  | Dict with as keys the categorical columns and as values the number of categories per categorical variable. |

**Returns**
- Type: `numpy.ndarray`

#### `loss_fn`

```python
loss_fn(pred_proba: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `pred_proba` | `numpy.ndarray` |  | Prediction probabilities of an instance. |
| `Y` | `numpy.ndarray` |  | One-hot representation of instance labels. |

**Returns**
- Type: `numpy.ndarray`

#### `reset_predictor`

```python
reset_predictor(predictor: Union[Callable, keras.src.models.model.Model]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Union[Callable, keras.src.models.model.Model]` |  | New predictor function/model. |

**Returns**
- Type: `None`

#### `score`

```python
score(X: numpy.ndarray, adv_class: int, orig_class: int, eps: float = 1e-10) -> float
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Instance to encode and calculate distance metrics for. |
| `adv_class` | `int` |  | Predicted class on the perturbed instance. |
| `orig_class` | `int` |  | Predicted class on the original instance. |
| `eps` | `float` | `1e-10` | Small number to avoid dividing by 0. |

**Returns**
- Type: `float`

## Functions
### `CounterFactualProto`

```python
CounterFactualProto(args, kwargs)
```

The class name `CounterFactualProto` is deprecated, please use `CounterfactualProto`.

# `alibi.explainers.cfrl_tabular`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_pytorch`
```python
has_pytorch: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `has_tensorflow`
```python
has_tensorflow: bool = True
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

## `ConcatTabularPostprocessing`

_Inherits from:_ `Postprocessing`, `ABC`

Tabular feature columns concatenation post-processing.

## `CounterfactualRLTabular`

_Inherits from:_ `CounterfactualRL`, `Explainer`, `FitMixin`, `ABC`, `Base`

Counterfactual Reinforcement Learning Tabular.

### Constructor

```python
CounterfactualRLTabular(self, predictor: Callable[[numpy.ndarray], numpy.ndarray], encoder: 'Union[tensorflow.keras.Model, torch.nn.Module]', decoder: 'Union[tensorflow.keras.Model, torch.nn.Module]', encoder_preprocessor: Callable, decoder_inv_preprocessor: Callable, coeff_sparsity: float, coeff_consistency: float, feature_names: List[str], category_map: Dict[int, List[str]], immutable_features: Optional[List[str]] = None, ranges: Optional[Dict[str, Tuple[int, int]]] = None, weight_num: float = 1.0, weight_cat: float = 1.0, latent_dim: Optional[int] = None, backend: str = 'tensorflow', seed: int = 0, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  | A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. For classification task, the second dimension of the output should match the number of classes. Thus, the output can be either a soft label distribution or a hard label distribution (i.e. one-hot encoding) without affecting the performance since `argmax` is applied to the predictor's output. |
| `encoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Pretrained heterogeneous encoder network. |
| `decoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  | Pretrained heterogeneous decoder network. The output of the decoder must be a list of tensors. |
| `encoder_preprocessor` | `Callable` |  | Auto-encoder data pre-processor. Depending on the input format, the pre-processor can normalize numerical attributes, transform label encoding to one-hot encoding etc. |
| `decoder_inv_preprocessor` | `Callable` |  | Auto-encoder data inverse pre-processor. This is the inverse function of the pre-processor. It can denormalize numerical attributes, transform one-hot encoding to label encoding, feature type casting etc. |
| `coeff_sparsity` | `float` |  | Sparsity loss coefficient. |
| `coeff_consistency` | `float` |  | Consistency loss coefficient. |
| `feature_names` | `List[str]` |  | List of feature names. This should be provided by the dataset. |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible values for a feature. This should be provided by the dataset. |
| `immutable_features` | `Optional[List[str]]` | `None` | List of immutable features. |
| `ranges` | `Optional[Dict[str, Tuple[int, int]]]` | `None` | Numerical feature ranges. Note that exist numerical features such as ``'Age'``, which are  allowed to increase only. We denote those by ``'inc_feat'``. Similarly, there exist features  allowed to decrease only. We denote them by ``'dec_feat'``. Finally, there are some free feature, which we denote by ``'free_feat'``. With the previous notation, we can define ``range = {'inc_feat': [0, 1], 'dec_feat': [-1, 0], 'free_feat': [-1, 1]}``. ``'free_feat'`` can be omitted, as any unspecified feature is considered free. Having the ranges of a feature `{'feat': [a_low, a_high}`, when sampling is performed the numerical value will be clipped between `[a_low * (max_val - min_val), a_high * [max_val - min_val]]`, where `a_low` and `a_high` are the minimum and maximum values the feature ``'feat'``. This implies that `a_low` and `a_high` are not restricted to ``{-1, 0}`` and ``{0, 1}``, but can be any float number in-between `[-1, 0]` and `[0, 1]`. |
| `weight_num` | `float` | `1.0` | Numerical loss weight. |
| `weight_cat` | `float` | `1.0` | Categorical loss weight. |
| `latent_dim` | `Optional[int]` | `None` | Auto-encoder latent dimension. Can be omitted if the actor network is user specified. |
| `backend` | `str` | `'tensorflow'` | Deep learning backend: ``'tensorflow'`` | ``'pytorch'``. Default ``'tensorflow'``. |
| `seed` | `int` | `0` | Seed for reproducibility. The results are not reproducible for ``'tensorflow'`` backend. |
| `**kwargs` |  |  | Used to replace any default parameter from :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS`. |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[List[Dict[str, List[Union[float, str]]]]] = None, batch_size: int = 100, diversity: bool = False, num_samples: int = 1, patience: int = 1000, tolerance: float = 0.001) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Input instances to generate counterfactuals for. |
| `Y_t` | `numpy.ndarray` |  | Target labels. |
| `C` | `Optional[List[Dict[str, List[Union[float, str]]]]]` | `None` | List of conditional dictionaries. If ``None``, it means that no conditioning was used during training (i.e. the `conditional_func` returns ``None``). If conditioning was used during training but no conditioning is desired for the current input, an empty list is expected. |
| `batch_size` | `int` | `100` | Batch size to use when generating counterfactuals. |
| `diversity` | `bool` | `False` | Whether to generate diverse counterfactual set for the given instance. Only supported for a single input instance. |
| `num_samples` | `int` | `1` | Number of diversity samples to be generated. Considered only if ``diversity=True``. |
| `patience` | `int` | `1000` | Maximum number of iterations to perform diversity search stops. If -1, the search stops only if the desired number of samples has been found. |
| `tolerance` | `float` | `0.001` | Tolerance to distinguish two counterfactual instances. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `fit`

```python
fit(X: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

## `SampleTabularPostprocessing`

_Inherits from:_ `Postprocessing`, `ABC`

Tabular sampling post-processing. Given the output of the heterogeneous auto-encoder the post-processing
functions samples the output according to the conditional vector. Note that the original input instance
is required to perform the conditional sampling.

### Constructor

```python
SampleTabularPostprocessing(self, category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `category_map` | `Dict[int, List[str]]` |  | Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible feature values. |
| `stats` | `Dict[int, Dict[str, float]]` |  | Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical feature in the training set. Each key is an index of the column and each value is another dictionary containing ``'min'`` and ``'max'`` keys. |

# `alibi.explainers.cfrl_tabular`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
### `has_pytorch`
```python
has_pytorch: bool = True
```
### `has_tensorflow`
```python
has_tensorflow: bool = True
```
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
| `predictor` | `Callable[[.[<class 'numpy.ndarray'>]], numpy.ndarray]` |  |  |
| `encoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `decoder` | `Union[tensorflow.keras.Model, torch.nn.Module]` |  |  |
| `encoder_preprocessor` | `Callable` |  |  |
| `decoder_inv_preprocessor` | `Callable` |  |  |
| `coeff_sparsity` | `float` |  |  |
| `coeff_consistency` | `float` |  |  |
| `feature_names` | `List[str]` |  |  |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `immutable_features` | `Optional[List[str]]` | `None` |  |
| `ranges` | `Optional[Dict[str, Tuple[int, int]]]` | `None` |  |
| `weight_num` | `float` | `1.0` |  |
| `weight_cat` | `float` | `1.0` |  |
| `latent_dim` | `Optional[int]` | `None` |  |
| `backend` | `str` | `'tensorflow'` |  |
| `seed` | `int` | `0` |  |

### Methods

#### `explain`

```python
explain(X: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[List[Dict[str, List[Union[float, str]]]]] = None, batch_size: int = 100, diversity: bool = False, num_samples: int = 1, patience: int = 1000, tolerance: float = 0.001) -> alibi.api.interfaces.Explanation
```

Computes counterfactuals for the given instances conditioned on the target and the conditional vector.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `Y_t` | `numpy.ndarray` |  |  |
| `C` | `Optional[List[Dict[str, List[Union[float, str]]]]]` | `None` |  |
| `batch_size` | `int` | `100` |  |
| `diversity` | `bool` | `False` |  |
| `num_samples` | `int` | `1` |  |
| `patience` | `int` | `1000` |  |
| `tolerance` | `float` | `0.001` |  |

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
| `category_map` | `Dict[int, List[str]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |

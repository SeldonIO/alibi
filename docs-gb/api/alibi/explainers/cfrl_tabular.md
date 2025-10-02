# `alibi.explainers.cfrl_tabular`
## Classes
### `ConcatTabularPostprocessing` (_inherits from `Postprocessing`, `ABC`)

Tabular feature columns concatenation post-processing.

#### Constructor

```python
ConcatTabularPostprocessing(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

### `CounterfactualRLTabular` (_inherits from `CounterfactualRL`, `Explainer`, `FitMixin`, `ABC`, `Base`)

Counterfactual Reinforcement Learning Tabular.

#### Constructor

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
| `kwargs` |  |  |  |

#### Methods

##### `explain`

```python
explain(X: numpy.ndarray, Y_t: numpy.ndarray, C: Optional[List[Dict[str, List[Union[float, str]]]]] = None, batch_size: int = 100, diversity: bool = False, num_samples: int = 1, patience: int = 1000, tolerance: float = 0.001) -> alibi.api.interfaces.Explanation
```

Computes counterfactuals for the given instances conditioned on the target and the conditional vector.

Parameters
----------
X
    Input instances to generate counterfactuals for.
Y_t
    Target labels.
C
    List of conditional dictionaries. If ``None``, it means that no conditioning was used during training
    (i.e. the `conditional_func` returns ``None``). If conditioning was used during training but no
    conditioning is desired for the current input, an empty list is expected.
diversity
    Whether to generate diverse counterfactual set for the given instance. Only supported for a single
    input instance.
num_samples
    Number of diversity samples to be generated. Considered only if ``diversity=True``.
batch_size
    Batch size to use when generating counterfactuals.
patience
    Maximum number of iterations to perform diversity search stops. If -1, the search stops only if
    the desired number of samples has been found.
tolerance
    Tolerance to distinguish two counterfactual instances.

Returns
-------
explanation
    `Explanation` object containing the counterfactual with additional metadata as attributes.             See usage `CFRL examples`_ for details.

    .. _CFRL examples:
        https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html

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

##### `fit`

```python
fit(X: numpy.ndarray) -> alibi.api.interfaces.Explainer
```

Fit the model agnostic counterfactual generator.

Parameters
----------
X
    Training data array.

Returns
-------
self
    The explainer itself.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

### `SampleTabularPostprocessing` (_inherits from `Postprocessing`, `ABC`)

Tabular sampling post-processing. Given the output of the heterogeneous auto-encoder the post-processing

functions samples the output according to the conditional vector. Note that the original input instance
is required to perform the conditional sampling.

#### Constructor

```python
SampleTabularPostprocessing(self, category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]])
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `category_map` | `Dict[int, List[str]]` |  |  |
| `stats` | `Dict[int, Dict[str, float]]` |  |  |

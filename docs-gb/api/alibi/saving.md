# `alibi.saving`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
### `NOT_SUPPORTED`
```python
NOT_SUPPORTED: list = ['DistributedAnchorTabular', 'CEM', 'Counterfactual', 'CounterfactualProto']
```
## `NumpyEncoder`

_Inherits from:_ `JSONEncoder`

### Methods

#### `default`

```python
default(obj)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `obj` |  |  |  |

## Functions
### `load_explainer`

```python
load_explainer(path: Union[str, os.PathLike], predictor) -> Explainer
```

Load an explainer from disk.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |
| `predictor` |  |  |  |

**Returns**
- Type: `Explainer`

### `save_explainer`

```python
save_explainer(explainer: Explainer, path: Union[str, os.PathLike]) -> None
```

Save an explainer to disk. Uses the `dill` module.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer` | `Explainer` |  |  |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

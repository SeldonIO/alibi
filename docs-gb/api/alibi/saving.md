# `alibi.saving`
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
----------
path
    Path to a directory containing the saved explainer.
predictor
    Model or prediction function used to originally initialize the explainer.

Returns
-------
An explainer instance.

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
----------
explainer
    Explainer instance to save to disk.
path
    Path to a directory. A new directory will be created if one does not exist.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer` | `Explainer` |  |  |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

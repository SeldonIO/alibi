# `alibi.saving`
## Constants
### `TYPE_CHECKING`
```python
TYPE_CHECKING: bool = False
```
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### `NOT_SUPPORTED`
```python
NOT_SUPPORTED: list = ['DistributedAnchorTabular', 'CEM', 'Counterfactual', 'CounterfactualProto']
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to a directory containing the saved explainer. |
| `predictor` |  |  | Model or prediction function used to originally initialize the explainer. |

**Returns**
- Type: `Explainer`

### `save_explainer`

```python
save_explainer(explainer: Explainer, path: Union[str, os.PathLike]) -> None
```

Save an explainer to disk. Uses the `dill` module.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer` | `Explainer` |  | Explainer instance to save to disk. |
| `path` | `Union[str, os.PathLike]` |  | Path to a directory. A new directory will be created if one does not exist. |

**Returns**
- Type: `None`

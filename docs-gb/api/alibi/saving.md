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

Extensible JSON <http://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
``.default()`` method with another method that returns a serializable
object for ``o`` if possible, otherwise it should call the superclass
implementation (to raise ``TypeError``).

### Constructor

```python
NumpyEncoder(self, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `skipkeys` |  | `False` |  |
| `ensure_ascii` |  | `True` |  |
| `check_circular` |  | `True` |  |
| `allow_nan` |  | `True` |  |
| `sort_keys` |  | `False` |  |
| `indent` |  | `None` |  |
| `separators` |  | `None` |  |
| `default` |  | `None` |  |

### Methods

#### `default`

```python
default(obj)
```

Implement this method in a subclass such that it returns

a serializable object for ``o``, or calls the base implementation
(to raise a ``TypeError``).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)

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

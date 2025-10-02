# `alibi.api.interfaces`
## Classes
### `AlibiPrettyPrinter` (_inherits from `PrettyPrinter`)

Overrides the built in dictionary pretty representation to look more similar to the external

prettyprinter libary.

#### Constructor

```python
AlibiPrettyPrinter(self, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

### `Base`

Base class for all `alibi` algorithms. Implements a structured approach to handle metadata.

#### Constructor

```python
Base(self, meta: dict = NOTHING) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` | `NOTHING` |  |

### `Explainer` (_inherits from `ABC`, `Base`)

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

#### Constructor

```python
Explainer(self, meta: dict = NOTHING) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` | `NOTHING` |  |

#### Methods

##### `explain`

```python
explain(X: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `load`

```python
load(path: Union[str, os.PathLike], predictor: typing.Any) -> alibi.api.interfaces.Explainer
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
| `predictor` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

##### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

Resets the predictor.

Parameters
----------
predictor
    New predictor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  |  |

**Returns**
- Type: `None`

##### `save`

```python
save(path: Union[str, os.PathLike]) -> None
```

Save an explainer to disk. Uses the `dill` module.

Parameters
----------
path
    Path to a directory. A new directory will be created if one does not exist.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

### `Explanation`

Explanation class returned by explainers.

#### Constructor

```python
Explanation(self, meta: dict, data: dict) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` |  |  |
| `data` | `dict` |  |  |

#### Methods

##### `from_json`

```python
from_json(jsonrepr) -> alibi.api.interfaces.Explanation
```

Create an instance of an `Explanation` class using a `json` representation of the `Explanation`.

Parameters
----------
jsonrepr
    `json` representation of an explanation.

Returns
-------
An Explanation object.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `jsonrepr` |  |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

##### `to_json`

```python
to_json() -> str
```

Serialize the explanation data and metadata into a `json` format.

Returns
-------
String containing `json` representation of the explanation.

**Returns**
- Type: `str`

### `FitMixin` (_inherits from `ABC`)

Helper class that provides a standard way to create an ABC using

inheritance.

#### Constructor

```python
FitMixin(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

#### Methods

##### `fit`

```python
fit(X: typing.Any) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

### `Summariser` (_inherits from `ABC`, `Base`)

Base class for prototype algorithms from :py:mod:`alibi.prototypes`.

#### Constructor

```python
Summariser(self, meta: dict = NOTHING) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` | `NOTHING` |  |

#### Methods

##### `load`

```python
load(path: Union[str, os.PathLike]) -> alibi.api.interfaces.Summariser
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Summariser`

##### `save`

```python
save(path: Union[str, os.PathLike]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

##### `summarise`

```python
summarise(num_prototypes: int) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `num_prototypes` | `int` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

## Functions
### `default_meta`

```python
default_meta() -> dict
```

**Returns**
- Type: `dict`

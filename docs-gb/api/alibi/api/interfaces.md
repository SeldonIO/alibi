# `alibi.api.interfaces`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.api.interfaces (WARNING)>
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

## `AlibiPrettyPrinter`

_Inherits from:_ `PrettyPrinter`

Overrides the built in dictionary pretty representation to look more similar to the external
prettyprinter libary.

### Constructor

```python
AlibiPrettyPrinter(self, *args, **kwargs)
```

## `Base`

Base class for all `alibi` algorithms. Implements a structured approach to handle metadata.

### Constructor

```python
Base(self, meta: dict = NOTHING) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` | `NOTHING` |  |

## `Explainer`

_Inherits from:_ `ABC`, `Base`

Base class for explainer algorithms from :py:mod:`alibi.explainers`.

### Methods

#### `explain`

```python
explain(X: typing.Any) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `load`

```python
load(path: Union[str, os.PathLike], predictor: typing.Any) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to a directory containing the saved explainer. |
| `predictor` | `typing.Any` |  | Model or prediction function used to originally initialize the explainer. |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

#### `reset_predictor`

```python
reset_predictor(predictor: typing.Any) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` | `typing.Any` |  | New predictor. |

**Returns**
- Type: `None`

#### `save`

```python
save(path: Union[str, os.PathLike]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  | Path to a directory. A new directory will be created if one does not exist. |

**Returns**
- Type: `None`

## `Explanation`

Explanation class returned by explainers.

### Constructor

```python
Explanation(self, meta: dict, data: dict) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `meta` | `dict` |  |  |
| `data` | `dict` |  |  |

### Methods

#### `from_json`

```python
from_json(jsonrepr) -> alibi.api.interfaces.Explanation
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `jsonrepr` |  |  | `json` representation of an explanation. |

**Returns**
- Type: `alibi.api.interfaces.Explanation`

#### `to_json`

```python
to_json() -> str
```

Serialize the explanation data and metadata into a `json` format.

**Returns**
- Type: `str`

## `FitMixin`

_Inherits from:_ `ABC`

### Methods

#### `fit`

```python
fit(X: typing.Any) -> alibi.api.interfaces.Explainer
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `typing.Any` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Explainer`

## `Summariser`

_Inherits from:_ `ABC`, `Base`

Base class for prototype algorithms from :py:mod:`alibi.prototypes`.

### Methods

#### `load`

```python
load(path: Union[str, os.PathLike]) -> alibi.api.interfaces.Summariser
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `alibi.api.interfaces.Summariser`

#### `save`

```python
save(path: Union[str, os.PathLike]) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `path` | `Union[str, os.PathLike]` |  |  |

**Returns**
- Type: `None`

#### `summarise`

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

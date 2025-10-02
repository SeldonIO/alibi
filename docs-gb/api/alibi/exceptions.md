# `alibi.exceptions`

This module defines the Alibi exception hierarchy and common exceptions
used across the library.

## Classes
### `AlibiException` (_inherits from `Exception`, `BaseException`, `ABC`)

Abstract base class of all alibi exceptions.

#### Constructor

```python
AlibiException(self, message: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

### `AlibiPredictorCallException`

#### Constructor

```python
AlibiPredictorCallException(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

### `AlibiPredictorReturnTypeError`

#### Constructor

```python
AlibiPredictorReturnTypeError(self, /, *args, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `args` |  |  |  |
| `kwargs` |  |  |  |

### `NotFittedError` (_inherits from `AlibiException`, `Exception`, `BaseException`, `ABC`)

This exception is raised whenever a compulsory call to a `fit` method has not been carried out.

#### Constructor

```python
NotFittedError(self, object_name: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  |  |

### `PredictorCallError` (_inherits from `AlibiException`, `Exception`, `BaseException`, `ABC`, `AlibiPredictorCallException`)

This exception is raised whenever a call to a user supplied predictor fails at runtime.

#### Constructor

```python
PredictorCallError(self, message: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

### `PredictorReturnTypeError` (_inherits from `AlibiException`, `Exception`, `BaseException`, `ABC`, `AlibiPredictorReturnTypeError`)

This exception is raised whenever the return type of a user supplied predictor is of

an unexpected or unsupported type.

#### Constructor

```python
PredictorReturnTypeError(self, message: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

### `SerializationError` (_inherits from `AlibiException`, `Exception`, `BaseException`, `ABC`)

This exception is raised whenever an explainer cannot be serialized.

#### Constructor

```python
SerializationError(self, message: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

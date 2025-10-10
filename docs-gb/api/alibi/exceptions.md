# `alibi.exceptions`

This module defines the Alibi exception hierarchy and common exceptions
used across the library.

## `AlibiException`

_Inherits from:_ `Exception`, `BaseException`, `ABC`

Abstract base class of all alibi exceptions.

### Constructor

```python
AlibiException(self, message: str) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

## `AlibiPredictorCallException`

### Constructor

```python
AlibiPredictorCallException(self, /, *args, **kwargs)
```

## `AlibiPredictorReturnTypeError`

### Constructor

```python
AlibiPredictorReturnTypeError(self, /, *args, **kwargs)
```

## `NotFittedError`

_Inherits from:_ `AlibiException`, `Exception`, `BaseException`, `ABC`

This exception is raised whenever a compulsory call to a `fit` method has not been carried out.

### Constructor

```python
NotFittedError(self, object_name: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  |  |

## `PredictorCallError`

_Inherits from:_ `AlibiException`, `Exception`, `BaseException`, `ABC`, `AlibiPredictorCallException`

This exception is raised whenever a call to a user supplied predictor fails at runtime.

## `PredictorReturnTypeError`

_Inherits from:_ `AlibiException`, `Exception`, `BaseException`, `ABC`, `AlibiPredictorReturnTypeError`

This exception is raised whenever the return type of a user supplied predictor is of
an unexpected or unsupported type.

## `SerializationError`

_Inherits from:_ `AlibiException`, `Exception`, `BaseException`, `ABC`

This exception is raised whenever an explainer cannot be serialized.

### Constructor

```python
SerializationError(self, message: str)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `message` | `str` |  |  |

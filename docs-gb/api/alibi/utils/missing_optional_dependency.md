# `alibi.utils.missing_optional_dependency`

Functionality for optional importing
This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.

## `MissingDependency`

Missing Dependency Class

Used to replace any object that requires unmet optional dependencies. Attribute access or calling the __call__
method on this object will raise an error.

### Constructor

```python
MissingDependency(self, object_name: str, err: Union[ModuleNotFoundError, ImportError], missing_dependency: str = 'all')
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  |  |
| `err` | `Union[ModuleNotFoundError, ImportError]` |  |  |
| `missing_dependency` | `str` | `'all'` |  |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `err_msg` | `` | Generate error message informing user to install missing dependencies. |

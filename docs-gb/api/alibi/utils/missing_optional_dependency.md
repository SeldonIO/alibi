# `alibi.utils.missing_optional_dependency`

Functionality for optional importing

This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.

## Constants
### `err_msg_template`
```python
err_msg_template: string.Template = <string.Template object at 0x16e56cbe0>
```
A string class for supporting $-substitutions.

### `ERROR_TYPES`
```python
ERROR_TYPES: dict = {'ray': 'ray', 'tensorflow': 'tensorflow', 'torch': 'torch', 'pytorch': 'torc...
```

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
| `object_name` | `str` |  | Name of object we are replacing |
| `err` | `Union[ModuleNotFoundError, ImportError]` |  | Error to be raised when the class is initialized or used |
| `missing_dependency` | `str` | `'all'` | Name of missing dependency required for object |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `err_msg` | `` | Generate error message informing user to install missing dependencies. |

## Functions
### `import_optional`

```python
import_optional(module_name: str, names: Optional[List[str]] = None) -> typing.Any
```

Import a module that depends on optional dependencies

Note: This function is used to import modules that depend on optional dependencies. Because it mirrors the python
import functionality its return type has to be `Any`. Using objects imported with this function can lead to
misspecification of types as `Any` when the developer intended to be more restrictive.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `module_name` | `str` |  | The module to import |
| `names` | `Optional[List[str]]` | `None` | The names to import from the module. If None, all names are imported. |

**Returns**
- Type: `typing.Any`

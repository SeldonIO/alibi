# `alibi.utils.missing_optional_dependency`

Functionality for optional importing
This module provides a way to import optional dependencies. In the case that the user imports some functionality from
alibi that is not usable due to missing optional dependencies this code is used to allow the import but replace it
with an object that throws an error on use. This way we avoid errors at import time that prevent the user using
functionality independent of the missing dependency.

## Classes
### `MissingDependency`

Missing Dependency Class

Used to replace any object that requires unmet optional dependencies. Attribute access or calling the __call__
method on this object will raise an error.

#### Constructor

```python
MissingDependency(self, object_name: str, err: Union[ModuleNotFoundError, ImportError], missing_dependency: str = 'all')
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `object_name` | `str` |  |  |
| `err` | `Union[ModuleNotFoundError, ImportError]` |  |  |
| `missing_dependency` | `str` | `'all'` |  |

#### Properties

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

Parameters
----------
module_name
    The module to import
names
    The names to import from the module. If None, all names are imported.

Returns
-------
The module or named objects within the modules if names is not None. If the import fails due to a
ModuleNotFoundError or ImportError then the requested module or named objects are replaced with instances of
the MissingDependency class above.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `module_name` | `str` |  |  |
| `names` | `Optional[List[str]]` | `None` |  |

**Returns**
- Type: `typing.Any`

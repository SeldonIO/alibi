# `alibi.utils.distributed`
## `ActorPool`

### Constructor

```python
ActorPool(self, actors)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `actors` |  |  |  |

### Methods

#### `get_next`

```python
get_next(timeout = None)
```

Returns the next pending result in order.

This returns the next result produced by :py:meth:`alibi.utils.distributed.ActorPool.submit`, blocking
for up to the specified timeout until it is available.

Returns

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `timeout` |  | `None` |  |

#### `get_next_unordered`

```python
get_next_unordered(timeout = None)
```

Returns any of the next pending results.

This returns some result produced by :py:meth:`alibi.utils.distributed.ActorPool.submit()`, blocking for up to
the specified timeout until it is available. Unlike :py:meth:`alibi.utils.distributed.ActorPool.get_next()`,
the results are not always returned in same order as submitted, which can improve performance.

Returns

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `timeout` |  | `None` |  |

#### `has_next`

```python
has_next()
```

Returns whether there are any pending results to return.

Returns

#### `map`

```python
map(fn, values, chunksize = 1)
```

Apply the given function in parallel over the `actors` and `values`. This returns an ordered iterator

that will return results of the map as they finish. Note that you must iterate over the iterator to force
the computation to finish.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` |  |  |  |
| `values` |  |  |  |
| `chunksize` |  | `1` |  |

#### `map_unordered`

```python
map_unordered(fn, values, chunksize = 1)
```

Similar to :py:meth:`alibi.utils.distributed.ActorPool.map`, but returning an unordered iterator.

This returns an unordered iterator that will return results of the map as they finish. This can be more
efficient that :py:meth:`alibi.utils.distributed.ActorPool.map` if some results take longer to compute
than others.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` |  |  |  |
| `values` |  |  |  |
| `chunksize` |  | `1` |  |

#### `submit`

```python
submit(fn: Callable, value: object)
```

Schedule a single task to run in the pool. This has the same argument semantics as

:py:meth:`alibi.utils.distributed.ActorPool.map`, but takes on a single value instead of a list of values.
The result can be retrieved using :py:meth:`alibi.utils.distributed.ActorPool.get_next()` /
:py:meth:`alibi.utils.distributed.ActorPool.get_next_unordered()`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` | `Callable` |  |  |
| `value` | `object` |  |  |

## `DistributedExplainer`

A class that orchestrates the execution of the execution of a batch of explanations in parallel.

### Constructor

```python
DistributedExplainer(self, distributed_opts: Dict[str, Any], explainer_type: Any, explainer_init_args: Tuple, explainer_init_kwargs: dict, concatenate_results: bool = True, return_generator: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  |  |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `Tuple` |  |  |
| `explainer_init_kwargs` | `dict` |  |  |
| `concatenate_results` | `bool` | `True` |  |
| `return_generator` | `bool` | `False` |  |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `actor_index` | `int` | Returns the index of the actor for which state is returned. |

### Methods

#### `create_parallel_pool`

```python
create_parallel_pool(explainer_type: typing.Any, explainer_init_args: Tuple, explainer_init_kwargs: dict)
```

Creates a pool of actors that can explain the rows of a dataset in parallel.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `Tuple` |  |  |
| `explainer_init_kwargs` | `dict` |  |  |

#### `get_explanation`

```python
get_explanation(X: numpy.ndarray, kwargs) -> Union[Generator[Tuple[int, typing.Any], None, None], List[typing.Any], typing.Any]
```

Performs distributed explanations of instances in `X`.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Union[Generator[Tuple[int, typing.Any], None, None], List[typing.Any], typing.Any]`

#### `return_attribute`

```python
return_attribute(name: str) -> typing.Any
```

Returns an attribute specified by its name. Used in a distributed context where the properties cannot be

accessed using the dot syntax.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `name` | `str` |  |  |

**Returns**
- Type: `typing.Any`

#### `set_actor_index`

```python
set_actor_index(value: int)
```

Sets actor index. This is used when the `DistributedExplainer` is in a separate process because `ray` does not

support calling property setters remotely

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `value` | `int` |  |  |

## `PoolCollection`

A wrapper object that turns a `DistributedExplainer` into a remote actor. This allows running multiple distributed

explainers in parallel.

### Constructor

```python
PoolCollection(self, distributed_opts: Dict[str, Any], explainer_type: Any, explainer_init_args: List[Tuple], explainer_init_kwargs: List[Dict], **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  |  |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `List[Tuple]` |  |  |
| `explainer_init_kwargs` | `List[Dict]` |  |  |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `remote_explainer_index` | `int` | Returns the index of the actor for which state is returned. |

### Methods

#### `create_explainer_handles`

```python
create_explainer_handles(distributed_opts: Dict[str, typing.Any], explainer_type: typing.Any, explainer_init_args: List[Tuple], explainer_init_kwargs: List[Dict], kwargs)
```

Creates multiple actors for `DistributedExplainer` so that tasks can be executed in parallel. The actors are

initialised with different arguments, so they represent different explainers.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  |  |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `List[Tuple]` |  |  |
| `explainer_init_kwargs` | `List[Dict]` |  |  |

#### `get_explanation`

```python
get_explanation(X, kwargs) -> List[Any]
```

Calls a collection of distributed explainers in parallel. Each distributed explainer will explain each row in

`X` in parallel.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` |  |  |  |

**Returns**
- Type: `List[Any]`

## `ResourceError`

_Inherits from:_ `Exception`, `BaseException`

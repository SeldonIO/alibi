# `alibi.utils.distributed`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.utils.distributed (WARNING)>
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

## `ActorPool`

### Constructor

```python
ActorPool(self, actors)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `actors` |  |  |  |

### Methods

### `get_next`

```python
get_next(timeout = None)
```

Returns the next pending result in order.

This returns the next result produced by :py:meth:`alibi.utils.distributed.ActorPool.submit`, blocking
for up to the specified timeout until it is available.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `timeout` |  | `None` |  |

### `get_next_unordered`

```python
get_next_unordered(timeout = None)
```

Returns any of the next pending results.

This returns some result produced by :py:meth:`alibi.utils.distributed.ActorPool.submit()`, blocking for up to
the specified timeout until it is available. Unlike :py:meth:`alibi.utils.distributed.ActorPool.get_next()`,
the results are not always returned in same order as submitted, which can improve performance.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `timeout` |  | `None` |  |

### `has_next`

```python
has_next()
```

Returns whether there are any pending results to return.

### `map`

```python
map(fn, values, chunksize = 1)
```

Apply the given function in parallel over the `actors` and `values`. This returns an ordered iterator

that will return results of the map as they finish. Note that you must iterate over the iterator to force
the computation to finish.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` |  |  |  |
| `values` |  |  |  |
| `chunksize` |  | `1` |  |

### `map_unordered`

```python
map_unordered(fn, values, chunksize = 1)
```

Similar to :py:meth:`alibi.utils.distributed.ActorPool.map`, but returning an unordered iterator.

This returns an unordered iterator that will return results of the map as they finish. This can be more
efficient that :py:meth:`alibi.utils.distributed.ActorPool.map` if some results take longer to compute
than others.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` |  |  |  |
| `values` |  |  |  |
| `chunksize` |  | `1` |  |

### `submit`

```python
submit(fn: Callable, value: object)
```

Schedule a single task to run in the pool. This has the same argument semantics as

:py:meth:`alibi.utils.distributed.ActorPool.map`, but takes on a single value instead of a list of values.
The result can be retrieved using :py:meth:`alibi.utils.distributed.ActorPool.get_next()` /
:py:meth:`alibi.utils.distributed.ActorPool.get_next_unordered()`.

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

### `create_parallel_pool`

```python
create_parallel_pool(explainer_type: typing.Any, explainer_init_args: Tuple, explainer_init_kwargs: dict)
```

Creates a pool of actors that can explain the rows of a dataset in parallel.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `Tuple` |  |  |
| `explainer_init_kwargs` | `dict` |  |  |

### `get_explanation`

```python
get_explanation(X: numpy.ndarray, kwargs) -> Union[Generator[Tuple[int, typing.Any], None, None], List[typing.Any], typing.Any]
```

Performs distributed explanations of instances in `X`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |

**Returns**
- Type: `Union[Generator[Tuple[int, typing.Any], None, None], List[typing.Any], typing.Any]`

### `return_attribute`

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

### `set_actor_index`

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

### `create_explainer_handles`

```python
create_explainer_handles(distributed_opts: Dict[str, typing.Any], explainer_type: typing.Any, explainer_init_args: List[Tuple], explainer_init_kwargs: List[Dict], kwargs)
```

Creates multiple actors for `DistributedExplainer` so that tasks can be executed in parallel. The actors are

initialised with different arguments, so they represent different explainers.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  |  |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `List[Tuple]` |  |  |
| `explainer_init_kwargs` | `List[Dict]` |  |  |

### `get_explanation`

```python
get_explanation(X, kwargs) -> List[Any]
```

Calls a collection of distributed explainers in parallel. Each distributed explainer will explain each row in

`X` in parallel.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` |  |  |  |

**Returns**
- Type: `List[Any]`

## `ResourceError`

_Inherits from:_ `Exception`, `BaseException`

Common base class for all non-exit exceptions.

### Constructor

```python
ResourceError(self, /, *args, **kwargs)
```

## Functions
### `batch`

```python
batch(X: numpy.ndarray, batch_size: Optional[int] = None, n_batches: int = 4) -> List[numpy.ndarray]
```

Splits the input into sub-arrays.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  |  |
| `batch_size` | `Optional[int]` | `None` |  |
| `n_batches` | `int` | `4` |  |

**Returns**
- Type: `List[numpy.ndarray]`

### `concatenate_minibatches`

```python
concatenate_minibatches(minibatch_results: Union[List[numpy.ndarray], List[List[numpy.ndarray]]]) -> Union[numpy.ndarray, List[numpy.ndarray]]
```

Merges the explanations computed on minibatches so that the distributed explainer returns the same output as the

sequential version. If the type returned by the explainer is not supported by the function, expand this function
by adding an appropriately named private function and use this function to check the input type and call it.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `minibatch_results` | `Union[List[numpy.ndarray], List[List[numpy.ndarray]]]` |  |  |

**Returns**
- Type: `Union[numpy.ndarray, List[numpy.ndarray]]`

### `default_target_fcn`

```python
default_target_fcn(actor: typing.Any, instances: tuple, kwargs: Optional[Dict] = None)
```

A target function that is executed in parallel given an actor pool. Its arguments must be an actor and a batch of

values to be processed by the actor. Its role is to execute distributed computations when an actor is available.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `actor` | `typing.Any` |  |  |
| `instances` | `tuple` |  |  |
| `kwargs` | `Optional[Dict]` | `None` |  |

### `invert_permutation`

```python
invert_permutation(p: list) -> numpy.ndarray
```

Inverts a permutation.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `list` |  |  |

**Returns**
- Type: `numpy.ndarray`

### `order_result`

```python
order_result(unordered_result: Generator[Tuple[int, typing.Any], None, None]) -> List[Any]
```

Re-orders the result of a distributed explainer so that the explanations follow the same order as the input to

the explainer.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `unordered_result` | `Generator[Tuple[int, typing.Any], None, None]` |  |  |

**Returns**
- Type: `List[Any]`

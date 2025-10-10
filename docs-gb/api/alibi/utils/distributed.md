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
| `actors` |  |  | List of `ray` actor handles to use in this pool. |

### Methods

#### `get_next`

```python
get_next(timeout = None)
```

Returns the next pending result in order.
This returns the next result produced by :py:meth:`alibi.utils.distributed.ActorPool.submit`, blocking
for up to the specified timeout until it is available.

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `timeout` |  | `None` |  |

#### `has_next`

```python
has_next()
```

Returns whether there are any pending results to return.

#### `map`

```python
map(fn, values, chunksize = 1)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` | `Callable` |  | Function that takes `(actor, value)` as argument and returns an `ObjectID` computing the result over the `value`. The `actor` will be considered busy until the `ObjectID` completes. |
| `values` | `list` |  | List of values that `fn(actor, value)` should be applied to. |
| `chunksize` | `int` | `1` | Splits the list of values to be submitted to the parallel process into sublists of size chunksize or less. |

#### `map_unordered`

```python
map_unordered(fn, values, chunksize = 1)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` | `Callable` |  | Function that takes `(actor, value)` as argument and returns an `ObjectID` computing the result over the `value`. The `actor` will be considered busy until the `ObjectID` completes. |
| `values` | `list` |  | List of values that `fn(actor, value)` should be applied to. |
| `chunksize` | `int` | `1` | Splits the list of values to be submitted to the parallel process into sublists of size chunksize or less. |

#### `submit`

```python
submit(fn: Callable, value: object)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `fn` | `Callable` |  | Function that takes `(actor, value)` as argument and returns an `ObjectID` computing the result over the `value`. The `actor` will be considered busy until the `ObjectID` completes. |
| `value` | `object` |  | Value to compute a result for. |

## `DistributedExplainer`

A class that orchestrates the execution of the execution of a batch of explanations in parallel.

### Constructor

```python
DistributedExplainer(self, distributed_opts: Dict[str, Any], explainer_type: Any, explainer_init_args: Tuple, explainer_init_kwargs: dict, concatenate_results: bool = True, return_generator: bool = False)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  | A dictionary with the following type (minimal signature):: class DistributedOpts(TypedDict): n_cpus: Optional[int] batch_size: Optional[int] The dictionary may contain two additional keys: - ``'actor_cpu_frac'`` : ``(float, <= 1.0, >0.0)`` - This is used to create more than one process                 on one CPU/GPU. This may not speed up CPU intensive tasks but it is worth experimenting with when                 few physical cores are available. In particular, this is highly useful when the user wants to share                 a GPU for multiple tasks, with the caviat that the machine learning framework itself needs to                 support running multiple replicas on the same GPU. See the `ray` documentation `here`_ for details. .. _here: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#fractional-resource-requirements - ``'algorithm'`` : ``str`` - this is specified internally by the caller. It is used in order to                 register target function callbacks for the parallel pool These should be implemented in the global                 scope. If not specified, its value will be ``'default'``, which will select a default target function                 which expects the actor has a `get_explanation` method. |
| `explainer_type` | `typing.Any` |  | Explainer class. |
| `explainer_init_args` | `Tuple` |  | Positional arguments to explainer constructor. |
| `explainer_init_kwargs` | `dict` |  | Keyword arguments to explainer constructor. |
| `concatenate_results` | `bool` | `True` | If ``True`` concatenates the results. See :py:func:`alibi.utils.distributed.concatenate_minibatches` for more details. |
| `return_generator` | `bool` | `False` | If ``True`` a generator that returns the results in the order the computation finishes is returned when `get_explanation` is called. Otherwise, the order of the results is the same as the order of the minibatches. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `actor_index` | `int` | Returns the index of the actor for which state is returned. |

### Methods

#### `create_parallel_pool`

```python
create_parallel_pool(explainer_type: typing.Any, explainer_init_args: Tuple, explainer_init_kwargs: dict)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `Tuple` |  |  |
| `explainer_init_kwargs` | `dict` |  |  |
| `See` | `constructor documentation.` |  |  |

#### `get_explanation`

```python
get_explanation(X: numpy.ndarray, kwargs) -> Union[Generator[Tuple[int, typing.Any], None, None], List[typing.Any], typing.Any]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | A batch of instances to be explained. Split into batches according to the settings passed to the constructor. |
| `**kwargs` |  |  | Any keyword-arguments for the explainer `explain` method. |

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
| `distributed_opts` | `, explainer_type, explainer_init_args, explainer_init_kwargs` |  | See :py:meth:`alibi.utils.distributed.DistributedExplainer` constructor documentation for explanations. Each entry in the list is a different explainer configuration (e.g., CEM in PN vs PP mode, different background dataset sizes for SHAP, etc). |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `List[Tuple]` |  |  |
| `explainer_init_kwargs` | `List[Dict]` |  |  |
| `**kwargs` |  |  | Any other kwargs, passed to the `DistributedExplainer` objects. |

### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `remote_explainer_index` | `int` | Returns the index of the actor for which state is returned. |

### Methods

#### `create_explainer_handles`

```python
create_explainer_handles(distributed_opts: Dict[str, typing.Any], explainer_type: typing.Any, explainer_init_args: List[Tuple], explainer_init_kwargs: List[Dict], kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `distributed_opts` | `Dict[str, typing.Any]` |  |  |
| `explainer_type` | `typing.Any` |  |  |
| `explainer_init_args` | `List[Tuple]` |  |  |
| `explainer_init_kwargs` | `List[Dict]` |  |  |
| `distributed_opts,` | `explainer_type, explainer_init_args, explainer_init_kwargs, **kwargs` |  | See :py:meth:`alibi.utils.distributed.PoolCollection`. |

#### `get_explanation`

```python
get_explanation(X, kwargs) -> List[Any]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` |  |  | Batch of instances to be explained. |

**Returns**
- Type: `List[Any]`

## `ResourceError`

_Inherits from:_ `Exception`, `BaseException`

## Functions
### `batch`

```python
batch(X: numpy.ndarray, batch_size: Optional[int] = None, n_batches: int = 4) -> List[numpy.ndarray]
```

Splits the input into sub-arrays.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `X` | `numpy.ndarray` |  | Array to be split. |
| `batch_size` | `Optional[int]` | `None` | The size of each batch. In particular - if `batch_size` is not ``None``, batches of this size are created. The sizes of the batches created might         vary if the 0-th dimension of `X` is not divisible by `batch_size`. For an array of length `l` that should         be split into `n` sections, it returns `l % n` sub-arrays of size `l//n + 1` and the rest of  `size l//n` - if `batch_size` is ``None``, then `X` is split into `n_batches` sub-arrays. |
| `n_batches` | `int` | `4` | Number of batches in which to split the sub-array. Only used if ``batch_size = None`` |

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
| `minibatch_results` | `Union[List[numpy.ndarray], List[List[numpy.ndarray]]]` |  | Explanations for each minibatch. |

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
| `actor` | `typing.Any` |  | A `ray` actor. This is typically a class decorated with the `@ray.remote decorator`, that has been subsequently instantiated using ``cls.remote(*args, **kwargs)``. |
| `instances` | `tuple` |  | A `(batch_index, batch)` tuple containing the batch of instances to be explained along with a batch index. |
| `kwargs` | `Optional[Dict]` | `None` | A list of keyword arguments for the actor `get_explanation` method. |

### `invert_permutation`

```python
invert_permutation(p: list) -> numpy.ndarray
```

Inverts a permutation.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `list` |  | Some permutation of `0, 1, ..., len(p)-1`. Returns an array `s`, where `s[i]` gives the index of `i` in `p`. |

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
| `unordered_result` | `Generator[Tuple[int, typing.Any], None, None]` |  | Each tuple contains the batch id as the first entry and the explanations for that batch as the second. |

**Returns**
- Type: `List[Any]`

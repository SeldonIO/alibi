# `alibi.datasets.tensorflow`
## Functions
### `fetch_fashion_mnist`

```python
fetch_fashion_mnist(return_X_y: bool = False) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Loads the Fashion MNIST dataset.

Parameters
----------
return_X_y:
    If ``True``, an `N x M x P` array of data points and `N`-array of labels are returned
    instead of a dict.

Returns
-------
If ``return_X_y=False``, a Bunch object with fields 'data', 'targets' and 'target_names'
is returned. Otherwise an array with data points and an array of labels is returned.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` |  |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

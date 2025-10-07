# `alibi.datasets.tensorflow`
## Functions
### `fetch_fashion_mnist`

```python
fetch_fashion_mnist(return_X_y: bool = False) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Loads the Fashion MNIST dataset.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` | If ``True``, an `N x M x P` array of data points and `N`-array of labels are returned instead of a dict. |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

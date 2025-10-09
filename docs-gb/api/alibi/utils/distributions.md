# `alibi.utils.distributions`
## Functions
### `kl_bernoulli`

```python
kl_bernoulli(p: numpy.ndarray, q: numpy.ndarray) -> numpy.ndarray
```

Compute KL-divergence between 2 probabilities `p` and `q`. `len(p)` divergences are calculated

simultaneously.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  |  |
| `q` | `numpy.ndarray` |  |  |

**Returns**
- Type: `numpy.ndarray`

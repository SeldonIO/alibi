# `alibi.utils.distributions`
## Functions
### `kl_bernoulli`

```python
kl_bernoulli(p: numpy.ndarray, q: numpy.ndarray) -> numpy.ndarray
```

Compute KL-divergence between 2 probabilities `p` and `q`. `len(p)` divergences are calculated

simultaneously.

Parameters
----------
p
    Probability.
q
    Probability.

Returns
-------
Array with the KL-divergence between `p` and `q`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `p` | `numpy.ndarray` |  | Probability. |
| `q` | `numpy.ndarray` |  | Probability. |

**Returns**
- Type: `numpy.ndarray`

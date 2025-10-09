# `alibi.utils.wrappers`
## `ArgmaxTransformer`

A transformer for converting classification output probability

tensors to class labels. It assumes the predictor is a callable
that can be called with a `N`-tensor of data points `x` and produces
an `N`-tensor of outputs.

### Constructor

```python
ArgmaxTransformer(self, predictor)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `predictor` |  |  |  |

## `Predictor`

### Constructor

```python
Predictor(self, clf, preprocessor=None)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `clf` |  |  |  |
| `preprocessor` |  | `None` |  |

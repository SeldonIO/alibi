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

## Functions
### `methdispatch`

```python
methdispatch(func)
```

A decorator that is used to support singledispatch style functionality
for instance methods. By default, singledispatch selects a function to
call from registered based on the type of args[0]::

    def wrapper(*args, **kw):
        return dispatch(args[0].__class__)(*args, **kw)

This uses singledispatch to do achieve this but instead uses `args[1]`
since `args[0]` will always be self.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `func` |  |  |  |

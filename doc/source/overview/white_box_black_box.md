# White-box and black-box models

Explainer algorithms can be categorised in many ways (see [this
table](algorithms.md#model-explanations)), but perhaps the most
fundamental one is whether they work with **white-box** or **black-box**
models.

**White-box** is a term used for any model that the explainer method can
“look inside” and manipulate arbitrarily. In the context of `alibi`
this category of models corresponds to Python objects that represent
models, for example instances of `sklearn.base.BaseEstimator`,
`tensorflow.keras.Model`, `torch.nn.Module` etc. The exact type of
the white-box model in question enables different white-box explainer
methods. For example, `tensorflow` and `torch` models are
*gradient-based* which enables *gradient-based* explainer methods such
as [Integrated Gradients](../methods/IntegratedGradients.ipynb)  [^id4]
whilst various types of tree-based models are supported by
[TreeShap](../methods/TreeSHAP.ipynb).

On the other hand, a **black-box** model describes any model that the
explainer method may not inspect and modify arbitrarily. The only
interaction with the model is via calling its `predict` function (or
similar) on data and receiving predictions back. In the context of
`alibi` black-box models have a concrete definition—they are functions
that take in a `numpy` array representing data and return a `numpy`
array representing a prediction. Using [type
hints](https://docs.python.org/3/library/typing.html) we can define a
general black-box model (also referred to as a *prediction function*) to
be of type `Callable[[np.ndarray], np.ndarray]` [^id5]. Explainers
that expect black-box models as input are very flexible as *any* type of
function that conforms to the expected type can be explained by
black-box explainers.

```{note}
In addition to the expected type, black-box models **must** be
compatible with batch prediction. I.e. `alibi` explainers assume
that the first dimension of the input array is always batch.
```

```{warning}
There is currently one exception to the black-box interface: the
`AnchorText` explainer expects the prediction function to be of
type `Callable[[List[str], np.ndarray]`, i.e. the model is expected
to work on batches of raw text (here `List[str]` indicates a batch
of text strings). See [this
example](../examples/anchor_text_movie.ipynb) for more
information.
```

## Wrapping white-box models into black-box models

Models in Python all start out as white-box models (i.e. custom Python
objects from some modelling library like `sklearn` or `tensorflow`).
However, to be used with explainers that expect a black-box prediction
function, the user has to define a prediction function that conforms to
the black-box definition given above. Here we give a few common examples
and some pointers about creating a black-box prediction function from a
white-box model. In what follows we distinguish between the original
white-box `model` and the wrapped black-box `predictor` function.

### Scikit-learn models

All `sklearn` models expose a `predict` method that already conforms
to the black-box function interface defined above which makes it easy to
create black-box predictors:

```python
predictor = model.predict
explainer = SomeExplainer(predictor, **kwargs)
```

In some cases for classifiers it may be more appropriate to expose the
`predict_proba` or `decision_function` method instead of
`predict`, see an example on [ALE for
classifiers](../examples/ale_classification.ipynb).

### Tensorflow models

Tensorflow models (specifically instances of `tensorflow.keras.Model`)
expose a `predict` method that takes in `numpy` arrays and returns
predictions as `numpy` arrays [^id6]:

```python
predictor = model.predict
explainer = SomeExplainer(predictor), **kwargs)
```

### Pytorch models

Pytorch models (specifically instances of `torch.nn.Module`) expect
and return instances of `torch.Tensor` from the `forward` method,
thus we need to do a bit more work to define the `predictor` black-box
function:

```python
model.eval()

@torch.no_grad()
def predictor(X: np.ndarray) -> np.ndarray:
    X = torch.as_tensor(X, dtype=dtype, device=device)
    return model.forward(X).cpu().numpy()
```

Note that there are a few differences with `tensorflow` models: 

- Ensure the model is in the evaluation mode (i.e., `model.eval()`) and that the mode does not change to training (i.e., `model.train()`) between consecutive calls to the explainer. Otherwise consider including `model.eval()` inside the `predictor` function.
- Decorate the `predictor` with `@torch.no_grad()` to avoid the computation and storage of the gradients which are not needed.
- Explicit conversion to a tensor with a specific `dtype`. Whilst
`tensorflow` handles this internally when `predict` is called, for
`torch` we need to do this manually. 
- Explicit device selection for the tensor. This is an important step as `numpy` arrays are limited to
cpu and if your model is on a gpu it will expect its input tensors to be
on a gpu.
- Explicit conversion of prediction tensor to `numpy`. We first send the output to the cpu and then transform into `numpy` array.

If you are using [Pytorch Lightning](https://www.pytorchlightning.ai)
to create `torch` models, then the `dtype` and `device` can be
retrieved as attributes of your `LightningModule`, see
[here](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

### General models

Given the above examples, the pattern for defining a black-box predictor
from a white-box model is clear: define a `predictor` function that
manipulates the inputs to and outputs from the underlying model in a way
that conforms to the black-box model interface `alibi` expects:

```python
def predictor(X: np.ndarray) -> np.ndarray:
    inp = transform_input(X)
    output = model(inp) # or call the model-specific prediction method
    output = transform_output(output)
    return output

explainer = SomeExplainer(predictor, **kwargs)
```

Here `transform_input` and `transform_output` are general
user-defined functions that appropriately transform the input `numpy`
arrays in a format that the model expects and transform the output
predictions into a `numpy` array so that `predictor` is an `alibi`
compatible black-box function.

[^id4]: At the time of writing `IntegratedGradients` only supports
    `tensorflow` models.

[^id5]: Note that this definition limits black-box models to be
    *single-input* and *single-output* which are what most black-box
    `alibi` explainers can handle. In the general case the definition
    can be extended to *multi-input* and *multi-output* models,
    i.e. taking in and/or returning multiple arrays.

[^id6]: This is in contrast to the `__call__` and `call` methods which
    expect and return `tensorflow.Tensor` objects. However, using
    `__call__` may be preferable for performance in [some
    cases](https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict)
    (this would require transforming inputs and outputs similar to the
    `torch` example).

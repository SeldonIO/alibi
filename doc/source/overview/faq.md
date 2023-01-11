# Frequently Asked Questions

## General troubleshooting

### I'm getting code errors using a method on my model and my data

There can be many reasons why the method does not work. For code exceptions it is a good idea to check the following:
 - Read the [docstrings](../api/modules.rst) of the method, paying close attention to the [type hints](https://docs.python.org/3/library/typing.html) as most errors are due to misformatted input arguments.
 - Check that your model signature (its type and expected inputs/outputs) are in the right format. Typically this means taking as input a `numpy` array representing a batch of data and returning a `numpy` array representing class labels, probabilities or regression values. For further details refer to [White-box and black-box models](../overview/white_box_black_box.md).
 - Check the expected input type for the `explain` method. Note that in many cases (e.g. all the Anchor methods) the `explain` method expects a single instance *without* a leading batch dimension, e.g. for `AnchorImage` a colour image of shape `(height, width, colour_channels)`.

### My model works on different input types, e.g. `pandas` dataframes instead of `numpy` arrays so the explainers don't work

At the time of writing we support models that operate on `numpy` arrays. You can write simple wrapper functions for your model so that it adheres to the format that `alibi` expects, [see here](../overview/white_box_black_box.md#wrapping-white-box-models-into-black-box-models). In the future we may support more diverse input types (see [#516](https://github.com/SeldonIO/alibi/issues/516)).

### Explanations are taking a long time to complete

Explanations can take different times as a function of the model, the data, and the explanation type itself. Refer to the [Introduction](../overview/high_level.md) and Methods sections for notes on algorithm complexity. You might need to experiment with the type of model, data points (specifically feature cardinality), and method parameters to ascertain if a specific method scales well for your use case.

### The explanation returned doesn't make sense to me

Explanations reflect the decision-making process of the model and not that of a biased human observer, see [Biases](../overview/high_level.md#biases). Moreover, depending on the method, the data, and the model, the explanations returned are valid but may be harder to interpret (e.g. see [Anchor explanations](#anchor-explanations) for some examples).

### Is there a way I can get more information from the library during the explanation generation process?

Yes! We use [Python logging](https://docs.python.org/3/howto/logging.html) to log info and debug messages from the library. You can configure logging for your script to see these messages during the execution of the code. Additionally, some methods also expose a `verbose` argument to print information to standard output. Configuring Python logging for your application will depend on your needs, but for simple scripts you can easily configure logging to print to standard error as follows:

```ipython3
import logging
logging.basicConfig(level=logging.DEBUG)
```
**Note:** this will display *all* logged messages with level `DEBUG` and higher from *all* libraries in use.

## Anchor explanations

### Why is my anchor explanation empty (tabular or text data) or black (image data)?

This is expected behaviour and signals that there is no salient subset of features that is necessary for the prediction to hold. In other words, with high probability (as measured by the precision), the predicted class of the data point does not change regardless of the perturbations applied to it.

**Note:** this behaviour can be typical for very imbalanced datasets, [see comment from the author](https://github.com/marcotcr/anchor/issues/71#issuecomment-863591122).

### Why is my anchor explanation so long (tabular or text data) or covers much of the image (image data)?

This is expected behaviour and can happen in two ways:
 - The data point to be explained lies near the decision boundary of the classifier. Thus, many more predicates are needed to ensure that a data point keeps the predicted class as small changes to the feature values may push the prediction to another class.
 - For tabular data, sampling perturbations is done using a training set. If the training set is imbalanced, explaining a minority class data point will result in oversampling perturbed features typical of majority classes so the algorithm would struggle to find a short anchor exceeding the specified precision level. For a concrete example, see [Anchor explanations for income prediction](../examples/anchor_tabular_adult.ipynb).

## Counterfactual explanations

### I'm using the methods [Counterfactual](../methods/CF.ipynb), [CounterfactualProto](../methods/CFProto.ipynb), or [CEM](../methods/CEM.ipynb) on a tree-based model such as decision trees, random forests,  or gradient boosted models (e.g. `xgboost`) but not finding any counterfactual examples

These methods only work on a subset of black-box models, namely ones whose decision function is differentiable with respect to the input data and hence amenable to gradient-based counterfactual search. Since for tree-based models, the decision function is piece-wise constant these methods won't work. It is recommended to use [CFRL](../methods/CFRL.ipynb) instead.

### I'm getting an error using the methods [Counterfactual](../methods/CF.ipynb), [CounterfactualProto](../methods/CFProto.ipynb), or [CEM](../methods/CEM.ipynb), especially if trying to use one of these methods together with [IntegratedGradients](../methods/IntegratedGradients.ipynb) or [CFRL](../methods/CFRL.ipynb)

At the moment the 3 counterfactual methods are implemented using TensorFlow 1.x constructs. This means that when running these methods, first we need to disable behaviour specific to TensorFlow 2.x as follows:

```ipython3
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
```
Unfortunately, running this line means it's impossible to run explainers based on TensorFlow 2.x such as [IntegratedGradients](../methods/IntegratedGradients.ipynb) or [CFRL](../methods/CFRL.ipynb). Thus at the moment, it is impossible to run these explainers together in the same Python interpreter session. Ultimately the fix is to rewrite the TensorFlow 1.x implementations in idiomatic TensorFlow 2.x and [some work has been done](https://github.com/SeldonIO/alibi/pull/403) but is currently not prioritised.

### Why am I'm unable to restrict the features allowed to changed in [CounterfactualProto](../methods/CFProto.ipynb)?

This is a known issue with the current implementation, see [here](https://github.com/SeldonIO/alibi/issues/327) and [here](https://github.com/SeldonIO/alibi/issues/366#issuecomment-820299804). It is currently blocked until we migrate the code to use TensorFlow 2.x constructs. In the meantime, it is recommended to use [CFRL](../methods/CFRL.ipynb) for counterfactual explanations with flexible feature-range constraints.


## Similarity explanations

### I'm using the [GradientSimilarity](../methods/Similarity.ipynb) method on a large model and it runs very slow. If I use `precompute_grads=True` I get out of memory errors. How do I solve this?

Large models with many parameters result in the similarity method running very slow and using `precompute_grads=True` may not be an option due to the memory cost. The best solutions for this problem are:

- Use the explainer on a reduced dataset. You can use [Prototype Methods](../methods/ProtoSelect.ipynb) to obtain a smaller representative sample.
- Freeze some parameters in the model so that when computing the gradients the simialrity method excludes them. If using [tensorflow](https://www.tensorflow.org/guide/keras/transfer_learning) you can do this by setting `trainable=False` on layers or specific parameters. For [pytorch](https://pytorch.org/docs/master/notes/autograd.html#locally-disabling-gradient-computation) we can set `requires_grad=False` on the relevent model parameters.

Note that doing so will cause the explainer to issue a warning on initialization, informing you there are non-trainable parameters in your model and the explainer will not use those when computng the similarity scores.

### I'm using the [GradientSimilarity](../methods/Similarity.ipynb) method on a tensorflow model and I keep getting warnings about non-trainable parameters but I haven't set any to be non-trainable?

This warning likely means that your model has layers that track statistics using non-trainable parameters such as batch normalization layers. The warning should list the specific tensors that are non-trainable so you should be able to check. If this is the case you don't need to worry as similarity methods don't use those parameters anyway. Otherwise you will see this warning if you have set one of the parameters to `trainable=False` and alibi is just making sure you know this is the case.

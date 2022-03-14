# Frequently Asked Questions

## General troubleshooting

**Q. I'm getting code errors using a method on my model and my data, what to do?**

**A.** There can be many reasons why the method does not work. For code exceptions it is a good idea to check the following:
 - Read the [docstrings](../api/modules.rst) of the method, paying close attention to the [type hints](https://docs.python.org/3/library/typing.html) as most errors are due to misformatted input arguments.
 - Check that your model signature (its type and expected inputs/outputs) are in the right format. For further details refer to [White-box and black-box models](../overview/white_box_black_box.md).

**Q. The explanations are taking a long time to complete, what to do?**

**A.** Explanations can take different times as a function of the model, the data, and the explanation type itself. Refer to the [Introduction](../overview/high_level.md) and Methods sections for notes on algorithm complexity. You might need to experiment with the type of model, data points (specifically feature cardinality), and method parameters to ascertain if a specific method scales well for your use case.

**Q. The explanation returned doesn't make sense to me, what to do?**

**A.** Explanations reflect the decision-making process of the model and not that of a biased human observer, see [Biases](../overview/high_level.md#biases). Moreover, depending on the method, the data, and the model, the explanations returned are valid but may be harder to interpret (e.g. see [Anchor explanations](#anchor-explanations) for some examples).

**Q. Is there a way I can get more information from the library during the explanation generation process?**

**A.** Yes! We use [Python logging](https://docs.python.org/3/howto/logging.html) to log debug messages from the library. You can configure logging for your script to see these messages during the execution of the code. Additionally, some methods also expose a `verbose` argument to print information to standard output.

## Anchor explanations

**Q. Why is my anchor explanation empty (tabular or text data) or black (image data)?**

**A.** This is expected behaviour and signals that there is no salient subset of features that is necessary for the prediction to hold. This can happen in two ways:
 - The predicted class of the data point does not change regardless of the perturbations applied to it.
 - The predicted class of the data point always changes regardless of the perturbations applied to it.

Note that this behaviour can be typical for very imbalanced datasets, [see comment from the author](https://github.com/marcotcr/anchor/issues/71#issuecomment-863591122).

**Q. Why is my anchor explanation so long (tabular or text data) or covers much of the image (image data)?**

**A.** This is expected behaviour and can happen in two ways:
 - The data point to be explained lies near the decision boundary of the classifier. Thus, many more predicates are needed to ensure that a data point keeps the predicted class as small changes to the feature values may push the prediction to another class.
 - For tabular data, sampling perturbations is done using a training set. If the training set is imbalanced, explaining a minority class data point will result in oversampling perturbed features typical of majority classes so the algorithm would struggle to find a short anchor exceeding the specified precision level. For a concrete example, see [Anchor explanations for income prediction](../examples/anchor_tabular_adult.ipynb).

## Counterfactual explanations

**Q. I'm trying to use the methods [Counterfactual](../methods/CF.ipynb), [CounterfactualProto](../methods/CFProto.ipynb), or [CEM](../methods/CEM.ipynb) on a tree-based model such as decision trees, random forests,  or gradient boosted models (e.g. `xgboost`) but not finding any counterfactual examples, what's wrong?**

**A.** These methods only work on a subset of black-box models, namely ones whose decision function is differentiable with respect to the input data and hence amenable to gradient-based counterfactual search. Since for tree-based models, the decision function is piece-wise constant these methods won't work. It is recommended to use [CFRL](../methods/CFRL.ipynb) instead.

**Q. I'm getting an error trying to use the methods [Counterfactual](../methods/CF.ipynb), [CounterfactualProto](../methods/CFProto.ipynb), or [CEM](../methods/CEM.ipynb), especially if trying to use one of these methods together with [IntegratedGradients](../methods/IntegratedGradients.ipynb) or [CFRL](../methods/CFRL.ipynb), what's happening?**

**A.** At the moment the 3 counterfactual methods are implemented using TensorFlow 1.x constructs. This means that when running these methods, first we need to disable behaviour specific to TensorFlow 2.x as follows:

```ipython3
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
```
Unfortunately, running this line means it's impossible to run explainers based on TensorFlow 2.x such as [IntegratedGradients](../methods/IntegratedGradients.ipynb) or [CFRL](../methods/CFRL.ipynb). Thus at the moment, it is impossible to run these explainers together in the same Python interpreter session. Ultimately the fix is to rewrite the TensorFlow 1.x implementations in idiomatic TensorFlow 2.x and [some work has been done](https://github.com/SeldonIO/alibi/pull/403) but is currently not prioritised.

**Q. Why am I'm unable to restrict the features allowed to changed in `CounterfactualProto`?**

**A.** This is a known issue with the current implementation, see [here](https://github.com/SeldonIO/alibi/issues/327) and [here](https://github.com/SeldonIO/alibi/issues/366#issuecomment-820299804). It is currently blocked until we migrate the code to use TensorFlow 2.x constructs. In the meantime, it is recommended to use [CFRL](../methods/CFRL.ipynb) for counterfactual explanations with flexible feature-range constraints.

# Algorithm overview

This page provides a high-level overview of the algorithms and their features currently implemented
in Alibi.

## Model Explanations
These algorithms provide **instance-specific** (sometimes also called **local**) explanations of ML model
predictions. Given a single instance and a model prediction they aim to answer the question "Why did
my model make this prediction?" Most of the following algorithms work with **black-box** models meaning that the
only requirement is to have access to a prediction function (which could be an API endpoint for a model in production).

The following table summarizes the capabilities of the current algorithms:

|Method|Models|Exp. types|Classification|Regression|Tabular|Text|Image|Cat. data|Train|Dist.|
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
|[ALE](../methods/ALE.html)|BB|global|✔|✔|✔| | | |✔| |
|[Anchors](../methods/Anchors.ipynb)|BB|local|✔| |✔|✔|✔|✔|For Tabular| |
|[CEM](../methods/CEM.ipynb)|BB* TF/Keras|local|✔| |✔| |✔| |Optional| |
|[Counterfactuals](../methods/CF.ipynb)|BB* TF/Keras|local|✔| |✔| |✔| |No| |
|[Prototype Counterfactuals](../methods/CFProto.ipynb)|BB* TF/Keras|local|✔| |✔| |✔|✔|Optional| |
|[Integrated Gradients](../methods/IntegratedGradients.ipynb)|TF/Keras|local|✔|✔|✔|✔|✔|✔|Optional| |
|[Kernel SHAP](../methods/KernelSHAP.ipynb)|BB|local  global|✔|✔|✔| | |✔|✔|✔|
|[Tree SHAP](../methods/TreeSHAP.ipynb)|WB|local  global|✔|✔|✔| | |✔|Optional| | |



Key:
 - **BB** - black-box (only require a prediction function)
 - **BB\*** - black-box but assume model is differentiable
 - **WB** - requires white-box model access. There may be limitations on models supported
 - **TF/Keras** - TensorFlow models via the Keras API
 - **Local** - instance specific explanation, why was this prediction made?
 - **Global** - explains the model with respect to a set of instances
 - **Cat. data** - support for categorical features
 - **Train** - whether a training set is required to fit the explainer
 - **Dist.** - whether a batch of explanations can be executed in parallel

**Accumulated Local Effects (ALE)**: calculates first-order feature effects on the model with
respect to a dataset. Intended for use on tabular datasets, currently supports numerical features.
[Documentation](../methods/ALE.ipynb), [regression example](../examples/ale_regression_boston.nblink),
[classification example](../examples/ale_classification.nblink).
 
**Anchor Explanations**: produce an "anchor" - a small subset of features and their ranges that will
almost always result in the same model prediction. [Documentation](../methods/Anchors.ipynb),
[tabular example](../examples/anchor_tabular_adult.nblink),
[text classification](../examples/anchor_text_movie.nblink),
[image classification](../examples/anchor_image_imagenet.nblink).

**Contrastive Explanation Method (CEM)**: produce a pertinent positive (PP) and a pertinent negative
(PN) instance. The PP instance finds the features that should be minimally and sufficiently present
to predict the same class as the original prediction (a PP acts as the "most compact" representation
of the instance to keep the same prediction). The PN instance identifies the features that should be
minimally and necessarily absent to maintain the original prediction (a PN acts as the closest
instance that would result in a different prediction). [Documentation](../methods/CEM.ipynb),
[tabular example](../examples/cem_iris.ipynb), [image classification](../examples/cem_mnist.ipynb).

**Counterfactual Explanations**: generate counterfactual examples using a simple loss function.
[Documentation](../methods/CF.ipynb), [image classification](../examples/cf_mnist.ipynb).

**Counterfactual Explanations Guided by Prototypes**: generate counterfactuals guided by nearest class
prototypes other than the class predicted on the original instance. It can use both an encoder or k-d trees
to define the prototypes. This method can speed up the search, especially for black box models, and create
interpretable counterfactuals. [Documentation](../methods/CFProto.ipynb),
[tabular example](../examples/cfproto_housing.nblink),
[tabular example with categorical features](../examples/cfproto_cat_adult_ohe.ipynb),
[image classification](../examples/cfproto_mnist.ipynb).

**Integrated gradients**: attribute an importance score to each element of the input or an internal layer of the the model  
with respect to a given baseline. The attributions are calculated as the path integral of the model gradients along a 
straight line from the baseline to the input.
[Documentation](../methods/IntegratedGradients.ipynb),
[MNIST example](../examples/integrated_gradients_mnist.nblink),
[Imagenet example](../examples/integrated_gradients_imagenet.nblink),
[IMDB example](../examples/integrated_gradients_imdb.nblink).

**Kernel Shapley Additive Explanations (Kernel SHAP)**: attribute the change of a model output with respect
to a given baseline (e.g., average over a reference set) to each of the input features. This is achieved for
each feature in turn, by averaging the difference in the model output observed when the feature whose contribution
is to be estimated is part of a group of "present" input features and the value observed when the feature is excluded
from said group. The features that are not "present" (i.e., are missing) are replaced with values from a background
dataset. This algorithm can be used to explain regression models and it is optimised to distribute batches of explanations.[Documentation](../methods/KernelSHAP.ipynb),
[continuous data](../examples/kernel_shap_wine_intro.ipynb),
[more continuous data](../examples/kernel_shap_wine_lr.ipynb),
[categorical data](../examples/kernel_shap_adult_lr.ipynb),
[distributed_batch_explanations](../examples/distributed_kernel_shap_adult_lr.ipynb)

**Tree Shapley Additive Explanations (Tree SHAP)**: attribute the change of a model output with respect to a baseline
(e.g., average over a reference set or inferred from node data) to each of the input features. Similar to Kernel SHAP,
the shap value of each feature is computed by averaging the difference of the model output observed when the feature
is part of a group of "present" features and when the feature is excluded from said group, over all possible subsets
of "present" features. Different estimation procedures for the effect of selecting different subsets of "present"
features on the model output give rise to the interventional feature perturbation and the path-dependent feature
perturbation variants of Tree SHAP. This algorithm can be used to explain regression models.
[Documentation](../methods/TreeSHAP.ipynb),
[interventional feature perturbation Tree SHAP](../examples/interventional_tree_shap_adult_xgb.ipynb),
[path-dependent feature perturbation Tree SHAP](../examples/path_dependent_tree_shap_adult_xgb.ipynb).

## Model Confidence
These algorithms provide **instance-specific** scores measuring the model confidence for making a
particular prediction.

|Method|Models|Classification|Regression|Tabular|Text|Images|Categorical Features|Train set required|
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
|[Trust Scores](../methods/TrustScores.ipynb)|BB|✔| |✔|✔[^1]|✔[^2]| |Yes|
|[Linearity Measure](../methods/LinearityMeasure.ipynb)|BB|✔|✔|✔| |✔| |Optional|

[^1]: depending on model
[^2]: may require dimensionality reduction

**Trust scores**: produce a "trust score" of a classifier's prediction. The trust score is the ratio
between the distance to the nearest class different from the predicted class and the distance to the
predicted class, higher scores correspond to more trustworthy predictions.
[Documentation](../methods/TrustScores.ipynb),
[tabular example](../examples/trustscore_iris.nblink),
[image classification](../examples/trustscore_mnist.nblink)

**Linearity measure**: produces a score quantifying how linear the model is around a test instance.
The linearity score measures the model linearity around a test instance by feeding the model linear
superpositions of inputs and comparing the outputs with the linear combination of outputs from
predictions on single inputs.
[Documentation](../methods/LinearityMeasure.ipynb)
[Tabular example](../examples/linearity_measure_iris.nblink),
[image classification](../examples/linearity_measure_fashion_mnist.nblink)

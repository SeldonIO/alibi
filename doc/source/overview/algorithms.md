# Algorithm overview

This page provides a high-level overview of the algorithms and their features currently implemented
in Alibi.

## Model Explanations
These algorithms provide **instance-specific** (sometimes also called **local**) explanations of ML model
predictions. Given a single instance and a model prediction they aim to answer the question "Why did
my model make this prediction?" The following algorithms all work with **black-box** models meaning that the
only requirement is to have acces to a prediction function (which could be an API endpoint for a model in production).
Note that local explanations can be combined to give insights into global model behaviour, but this comes at the
expense of significant runtime increase.

The following table summarizes the capabilities of the current algorithms:

|Explainer|Model types|Classification|Categorical data|Tabular|Text|Images|Need training set|
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
|[Anchors](../methods/Anchors.ipynb)|black-box|✔|✔|✔|✔|✔|For Tabular|
|[CEM](../methods/CEM.ipynb)|black-box, TF/Keras|✔|✘|✔|✘|✔|Optional|
|[Counterfactual Instances](../methods/CF.ipynb)|black-box, TF/Keras|✔|✘|✔|✘|✔|No|
|[Kernel SHAP](../methods/KernelSHAP.ipynb)|black-box|✔|✔|✔|✘|✘|✔|
|[Prototype Counterfactuals](../methods/CFProto.ipynb)|black-box, TF/Keras|✔|✔|✔|✘|✔|Optional|

**Anchor explanations**: produce an "anchor" - a small subset of features and their ranges that will
almost always result in the same model prediction. [Documentation](../methods/Anchors.ipynb),
[tabular example](../examples/anchor_tabular_adult.nblink),
[text classification](../examples/anchor_text_movie.nblink),
[image classification](../examples/anchor_image_imagenet.nblink).

**Contrastive explanation method (CEM)**: produce a pertinent positive (PP) and a pertinent negative
(PN) instance. The PP instance finds the features that should be minimally and sufficiently present
to predict the same class as the original prediction (a PP acts as the "most compact" representation
of the instance to keep the same prediction). The PN instance identifies the features that should be
minimally and necessarily absent to maintain the original prediction (a PN acts as the closest
instance that would result in a different prediction). [Documentation](../methods/CEM.ipynb),
[tabular example](../examples/cem_iris.ipynb), [image classification](../examples/cem_mnist.ipynb).

**Counterfactual instances**: generate counterfactual examples using a simple loss function. [Documentation](../methods/CF.ipynb), [image classification](../examples/cf_mnist.ipynb).

**Kernel Shapley Additive Explanation (SHAP)**: attribute the change of a model output with respect to a given baseline (e.g., average over a training set) to each of the model features. This is achieved for each feature in turn, by averaging the difference in the model output observed when excluding a feature from the input. The exclusion of a feature is achieved by replacing it with values from the background dataset. [Documentation](../methods/KernelSHAP.ipynb), [continuous data](../examples/kernel_shap_wine_intro.ipynb), [more continous_data](../examples/kernel_shap_wine_lr.ipynb), [categorical data](../examples/kernel_shap_adult_lr.ipynb).

**Prototype Counterfactuals**: generate counterfactuals guided by nearest class prototypes other than the class predicted on the original instance. It can use both an encoder or k-d trees to define the prototypes. This method can speed up the search, especially for black box models, and create interpretable counterfactuals. [Documentation](../methods/CFProto.ipynb), [tabular example](../examples/cfproto_housing.nblink), [tabular example with categorical features](../examples/cfproto_cat_adult_ohe.ipynb), [image classification](../examples/cfproto_mnist.ipynb).


## Model Confidence
These algorihtms provide **instance-specific** scores measuring the model confidence for making a
particular prediction.

|Algorithm|Model types|Classification|Regression|Categorical data|Tabular|Text|Images|Need training set|
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
|[Trust Scores](../methods/TrustScores.ipynb)|black-box|✔|✘|✘|✔|✔[^1]|✔[^2]|Yes|
|[Linearity Measure](../examples/linearity_measure_iris.ipynb)|black-box|✔|✔|✘|✔|✘|✔|Optional|

**Trust scores**: produce a "trust score" of a classifier's prediction. The trust score is the ratio
between the distance to the nearest class different from the predicted class and the distance to the
predicted class, higher scores correspond to more trustworthy predictions.
[Documentation](../methods/TrustScores.ipynb),
[tabular example](../examples/trustscore_iris.nblink),
[image classification](../examples/trustscore_mnist.nblink)

[^1]: Depending on model
[^2]: May require dimensionality reduction

**Linearity measure**: produces a score quantifying how linear the model is around a test instance.
The linearity score measures the model linearity around a test instance by feeding the model linear
superpositions of inputs and comparing the outputs with the linear combination of outputs from
predictions on single inputs.
[Tabular example](../examples/linearity_measure_iris.nblink),
[image classification](../examples/linearity_measure_fashion_mnist.nblink)

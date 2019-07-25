# Algorithm overview

This page provides a high-level overview of the algorithms and their features currently implemented
in Alibi.

## Model Explanations
These algorithms provide instance-specific (sometimes also called "local") explanations of ML model
predictions. Given a single instance and a model prediction they aim to answer the question "Why did
my model make this prediction?" The following table summarizes the capabilities of the current
algorithms:

|Explainer|Classification|Regression|Categorical features|Tabular|Text|Images|Needs training set|
|---|---|---|---|---|
|[Anchors](../methods/Anchors.ipynb)|✔|✘|✔|✔|✔|✔|For Tabular|
|[CEM](../methods/CEM.ipynb)|✔|✘|✘|✔|✘|✔|Optional|
|[Counterfactual Instances](../methods/CF.ipynb)|✔|✘|✘|✔|✘|✔|No|
|[Prototype Counterfactuals](../methods/CFProto.ipynb)|✔|✘|✘|✔|✘|✔|Optional|

**Anchor explanations**: produce an "anchor" - a small subset of features and their ranges that will
almost always result in the same model prediction. [Documentation](../methods/Anchors.ipynb),
[tabular example](../examples/anchor_tabular_adult.nblink),
[text classification](../examples/anchor_text_movie.nblink),
[image classification](../examples/anchor_image_imagenet.nblink).

**Contrastive explanation method (CEM)**: produce a pertinent positive (PP) and a pertinent negative
(PN) instance. The PP instance finds the features that should me minimally and sufficiently present
to predict the same class as the original prediction (a PP acts as the "most compact" representation
of the instance to keep the same prediction). The PN instance identifies the features that should be
minimally and necessarily absent to maintain the original prediction (a PN acts as the closest
instance that would result in a different prediction). [Documentation](../methods/CEM.ipynb),
[tabular example](../examples/cem_iris.ipynb), [image classification](../examples/cem_mnist.ipynb).

**Counterfactual instances**: generate counterfactual examples using a simple loss function. [Documentation](../methods/CF.ipynb), [image classification](../examples/cf_mnist.ipynb).

**Prototype Counterfactuals**: generate counterfactuals guided by nearest class prototypes other than the class predicted on the original instance. It can use both an encoder or k-d trees to define the prototypes. This method can speed up the search, especially for black box models, and create interpretable counterfactuals. [Documentation](../methods/CFProto.ipynb), [tabular example](../examples/cfproto_housing.nblink), [image classification](../examples/cfproto_mnist.ipynb).


## Model Confidence
These algorihtms provide instance-specific scores measuring the model confidence for making a
particular prediction.

|Algorithm|Classification|Regression|Categorical features|Tabular|Text|Images|Needs training set|
|---|---|---|---|---|
|[Trust Scores](../methods/TrustScores.ipynb)|✔|✘|✘|✔|✔[^1]|✔[^2]|Yes|
|[Linearity Measure](../examples/linearity_measure.ipynb)|✔|✔|✘|✔|✘|✔|Optional|

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
[Examples](../examples/linearity_measure.nblink)
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


## Model Confidence
These algorihtms provide instance-specific scores measuring the model confidence for making a
particular prediction.

|Algorithm|Classification|Regression|Categorical features|Tabular|Text|Images|Needs training set|
|---|---|---|---|---|
|[Trust Scores](../methods/Trust\ Scores.ipynb)|✔|✘|✘|✔|✔[^1]|✔[^2]|Yes|

**Trust scores**: produce a "trust score" of a classifier's prediction. The trust score is the ratio
between the distance to the nearest class different from the predicted class and the distance to the
predicted class, higher scores correspond to more trustworthy predictions.
[Documentation](../methods/Trust\ Scores.ipynb),
[tabular example](../examples/trustscore_iris.nblink).

[^1]: Depending on model
[^2]: May require dimensionality reduction

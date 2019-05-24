# Getting Started

## Installation
Alibi works with Python 3.5+ and can be installed from [PyPI](https://pypi.org/project/alibi):
```bash
pip install alibi
```

## Features
Alibi is a Python package designed to help explain the predictions of machine learning models, gauge
the confidence of predictions and eventually support wider capabilities of inspecting the
performance of models with respect to concept drift and algorithmic bias. The focus of the library
is to support the widest range of models using black-box methods where possible.

To get a list of the latest available model explanation algorithms, you can type:
```python
import alibi
alibi.explainers.__all__
```
```
['AnchorTabular', 'AnchorText', 'AnchorImage', 'CEM'] 
```

For gauging model confidence:
```python
alibi.confidence.__all__
```
```
['TrustScore']
```



For detailed information on the methods:
*  [Overview of available methods](../overview/algorithms.md)
    * [Anchor explanations](../methods/Anchors.ipynb)
    * [Contrastive Explanation Method (CEM)](../methods/CEM.ipynb)
    * [Counterfactuals Guided by Prototypes](../methods/CFProto.ipynb)
    * [Trust Scores](../methods/TrustScores.ipynb)

## Basic Usage
We will use the [Anchor method on tabular data](../methods/Anchors.ipynb#Tabular-Data) to illustrate
the usage of explainers in Alibi.

First, we import the explainer:
```python
from alibi.explainers import AnchorTabular
```
Next, we initialize it by passing it a prediction function and any other necessary arguments:
```python
explainer = AnchorTabular(predict_fn, feature_names)
```
Some methods require an additional `.fit` step which requires access to the training set the model
was trained on:
```python
explainer.fit(X_train)
```
Finally, we can call the explainer on a test instance which will return a dictionary containing the
explanation and any additional metadata returned by the computation:
```python
 explainer.explain(x)
```
The exact details will vary slightly from method to method, so we encourage the reader to become
familiar with the [types of algorithms supported](../overview/algorithms.md) in Alibi.
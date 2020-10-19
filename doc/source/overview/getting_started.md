# Getting Started

## Installation
Alibi works with Python 3.6+ and can be installed from [PyPI](https://pypi.org/project/alibi):
```bash
pip install alibi
```
Alternatively, the development version can be installed:
```bash
pip install git+https://github.com/SeldonIO/alibi.git 
```

## Features
Alibi is a Python package designed to help explain the predictions of machine learning models and gauge
the confidence of predictions. The focus of the library is to support the widest range of models using
black-box methods where possible.

To get a list of the latest available model explanation algorithms, you can type:
```python
import alibi
alibi.explainers.__all__
```
```
['ALE',
 'AnchorTabular',
 'AnchorText',
 'AnchorImage',
 'CEM',
 'CounterFactual',
 'CounterFactualProto'
 'KernelShap',
 'plot_ale'] 
```

For gauging model confidence:
```python
alibi.confidence.__all__
```
```
['linearity_measure',
 'LinearityMeasure',
 'TrustScore']
```



For detailed information on the methods:
*  [Overview of available methods](../overview/algorithms.md)
    * [Accumulated Local Effects](../methods/ALE.ipynb)
    * [Anchor explanations](../methods/Anchors.ipynb)
    * [Contrastive Explanation Method (CEM)](../methods/CEM.ipynb)
    * [Counterfactual Instances](../methods/CF.ipynb)
    * [Counterfactuals Guided by Prototypes](../methods/CFProto.ipynb)
    * [Kernel SHAP](../methods/KernelSHAP.ipynb)
    * [Integrated gradients](../methods/IntegratedGradients.ipynb)
    * [Linearity Measure](../methods/LinearityMeasure.ipynb)
    * [Trust Scores](../methods/TrustScores.ipynb)

## Basic Usage
The alibi explanation API takes inspiration from `scikit-learn`, consisting of distinct initialize,
fit and explain steps. We will use the [Anchor method on tabular data](../methods/Anchors.ipynb#Tabular-Data)
to illustrate the API.

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
```
AnchorTabular(meta={
    'name': 'AnchorTabular',
    'type': ['blackbox'],
    'explanations': ['local'],
    'params': {'seed': None, 'disc_perc': (25, 50, 75)}
})
```

Finally, we can call the explainer on a test instance which will return an `Explanation` object containing the
explanation and any additional metadata returned by the computation:
```python
 explanation = explainer.explain(x)
```

The returned `Explanation` object has `meta` and `data` attributes which are dictionaries containing any explanation
metadata (e.g. parameters, type of explanation) and the explanation itself respectively:

```python
explanation.meta
```
```
{'name': 'AnchorTabular',
 'type': ['blackbox'],
 'explanations': ['local'],
 'params': {'seed': None,
  'disc_perc': (25, 50, 75),
  'threshold': 0.95,
  'delta': ...truncated output...
```

```python
explanation.data
```
```
{'anchor': ['petal width (cm) > 1.80', 'sepal width (cm) <= 2.80'],
 'precision': 0.9839228295819936,
 'coverage': 0.31724137931034485,
 'raw': {'feature': [3, 1],
  'mean': [0.6453362255965293, 0.9839228295819936],
  'precision': [0.6453362255965293, 0.9839228295819936],
  'coverage': [0.20689655172413793, 0.31724137931034485],
  'examples': ...truncated output...
```

The top level keys of both `meta` and `data` dictionaries are also exposed as attributes for ease of use of the explanation:
```python
explanation.anchor
```
```
['petal width (cm) > 1.80', 'sepal width (cm) <= 2.80']
```

Some algorithms, such as [Kernel SHAP](../methods/KernelSHAP.ipynb), can run batches of explanations in parallel, if the number of cores is specified in the algorithm constructor:
```python
distributed_ks = KernelShap(predict_fn, distributed_opts={'n_cpus': 10})
```

Note that this requires the user to run ``pip install alibi[ray]`` to install dependencies of the distributed backend.

The exact details will vary slightly from method to method, so we encourage the reader to become
familiar with the [types of algorithms supported](../overview/algorithms.md) in Alibi.

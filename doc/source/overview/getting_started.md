# Getting Started

## Installation
Alibi can be installed from [PyPI](https://pypi.org/project/alibi):
```bash
pip install alibi
```

## Available methods
To get a list of the latest available model explanation algorithms, you can type:
```python
import alibi.explainers
alibi.explainers.__all__
```
<div class="highlight"><pre>
['AnchorTabular', 'AnchorText', 'AnchorImage', 'CEM'] 
</pre></div>

For detailed information on the methods:
*  [Overview of available methods](../overview/algorithms.md)
    * [Anchor explanations](../methods/Anchors.ipynb)
    * [Contrastive Explanation Method (CEM)](../methods/CEM.ipynb)

## Usage
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
Finally, we can call the explainer on a test instance which will return a dictionary containint the
explanation and any additional metadata returned by the computation:
```python
 explainer.explain(x)
```
The exact details will vary slightly from method to method, so we encourage the reader to become
familiar with the types of algorithms supported in Alibi.
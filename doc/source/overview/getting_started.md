# Getting Started

## Installation
Alibi works with Python 3.7+ and can be installed from [PyPI](https://pypi.org/project/alibi/) or [conda-forge](https://conda-forge.org/)
by following the instructions below.

``````{dropdown} Install via PyPI
```{div} sd-mb-3
- Alibi can be installed from [PyPI](https://pypi.org/project/alibi/) with `pip`:
```

`````{tab-set}

````{tab-item} Standard
:sync: label-standard
:class-label: sd-pt-0
```{div} sd-mb-1
Default installation.
```
```bash
pip install alibi
```
````

````{tab-item} SHAP
:sync: label-shap
:class-label: sd-pt-0
```{div} sd-mb-1
Installation with support for computing [SHAP](https://shap.readthedocs.io/en/stable/index.html) values.
```
```bash
pip install alibi[shap]
```
````

````{tab-item} Distributed
:class-label: sd-pt-0
:sync: label-dist
```{div} sd-mb-1
Installation with support for 
[distributed Kernel SHAP](../examples/distributed_kernel_shap_adult_lr.ipynb).
```
```bash
pip install alibi[ray]
```
````

````{tab-item} TensorFlow
:class-label: sd-pt-0
:sync: label-tensorflow
```{div} sd-mb-1
Installation with support for tensorflow backends. Required for 
- [Contrastive Explanation Method (CEM)](../methods/CEM.ipynb) 
- [Counterfactuals Guided by Prototypes](../methods/CFProto.ipynb) 
- [Counterfactual Instances](../methods/CF.ipynb)
- [Integrated gradients](../methods/IntegratedGradients.ipynb) 
- [Anchors on Textual data](../examples/anchor_text_movie.ipynb) with `sampling_strategy='language_model'` 
- One of Torch or TensorFlow is required for the [Counterfactuals with RL](../methods/CFRL.ipynb) methods
```
```bash
pip install alibi[tensorflow]
```
````

````{tab-item} Torch
:class-label: sd-pt-0
:sync: label-torch
```{div} sd-mb-1
Installation with support for torch backends. One of Torch or TensorFlow is required for the 
[Counterfactuals with RL](../methods/CFRL.ipynb) methods.
```
```bash
pip install alibi[torch]
```
````

````{tab-item} All
:class-label: sd-pt-0
:sync: label-all
```{div} sd-mb-1
Installs all optional dependencies.
```
```bash
pip install alibi[all]
```
````
`````
``````

``````{dropdown} Install via conda-forge
```{div} sd-mb-3
- To install the conda-forge version it is recommended to use [mamba](https://mamba.readthedocs.io/en/stable/), 
which can be installed to the *base* conda enviroment with:
```
```bash
conda install mamba -n base -c conda-forge
```
```{div} sd-mb-3
- `mamba` can then be used to install alibi in a conda enviroment:
```

`````{tab-set}

````{tab-item} Standard
:sync: label-standard
:class-label: sd-pt-0
```{div} sd-mb-1
Default installation.
```
```bash
mamba install -c conda-forge alibi
```
````

````{tab-item} SHAP
:sync: label-shap
:class-label: sd-pt-0
```{div} sd-mb-1
Installation with support for computing [SHAP](https://shap.readthedocs.io/en/stable/index.html) values.
```
```bash
mamba install -c conda-forge alibi shap
```
````

````{tab-item} Distributed
:sync: label-dist
:class-label: sd-pt-0
```{div} sd-mb-1
Installation with support for distributed computation of explanations.
```
```bash
mamba install -c conda-forge alibi ray 
```
````

`````
``````

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
'DistributedAnchorTabular', 
'AnchorText', 
'AnchorImage', 
'CEM', 
'Counterfactual', 
'CounterfactualProto', 
'CounterfactualRL', 
'CounterfactualRLTabular',
'PartialDependence',
'TreePartialDependence',
'PartialDependenceVariance',
'PermutationImportance',
'plot_ale',
'plot_pd',
'plot_pd_variance',
'plot_permutation_importance',
'IntegratedGradients', 
'KernelShap', 
'TreeShap',
'GradientSimilarity']
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

For dataset summarization
```python
alibi.prototypes.__all__
```
```
['ProtoSelect',
 'visualize_image_prototypes']
```


For detailed information on the methods:
*  [Overview of available methods](../overview/algorithms.md)
    * [Accumulated Local Effects](../methods/ALE.ipynb)
    * [Anchor explanations](../methods/Anchors.ipynb)
    * [Contrastive Explanation Method (CEM)](../methods/CEM.ipynb)
    * [Counterfactual Instances](../methods/CF.ipynb)
    * [Counterfactuals Guided by Prototypes](../methods/CFProto.ipynb)
    * [Counterfactuals with RL](../methods/CFRL.ipynb)
    * [Integrated gradients](../methods/IntegratedGradients.ipynb)
    * [Kernel SHAP](../methods/KernelSHAP.ipynb)
    * [Linearity Measure](../methods/LinearityMeasure.ipynb)
    * [ProtoSelect](../methods/ProtoSelect.ipynb)
    * [PartialDependence](../methods/PartialDependence.ipynb)
    * [PD Variance](../methods/PartialDependenceVariance.ipynb)
    * [Permutation Importance](../methods/PermutationImportance.ipynb)
    * [TreeShap](../methods/TreeSHAP.ipynb)
    * [Trust Scores](../methods/TrustScores.ipynb)
    * [Similarity explanations](../methods/Similarity.ipynb)

## Basic Usage
The alibi explanation API takes inspiration from `scikit-learn`, consisting of distinct initialize,
fit and explain steps. We will use the [Anchor method on tabular data](/methods/Anchors.ipynb#Tabular-Data)
to illustrate the API.

First, we import the explainer:
```python
from alibi.explainers import AnchorTabular
```
Next, we initialize it by passing it a [prediction function](white_box_black_box.md) and any other necessary arguments:
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

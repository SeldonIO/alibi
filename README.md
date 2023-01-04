<p align="center">
  <img src="https://raw.githubusercontent.com/SeldonIO/alibi/master/doc/source/_static/Alibi_Explain_Logo_rgb.png" alt="Alibi Logo" width="50%">
</p>

<!--- BADGES: START --->

[![Build Status](https://github.com/SeldonIO/alibi-detect/workflows/CI/badge.svg?branch=master)][#build-status]
[![Documentation Status](https://readthedocs.org/projects/alibi/badge/?version=latest)][#docs-package]
[![codecov](https://codecov.io/gh/SeldonIO/alibi/branch/master/graph/badge.svg)](https://codecov.io/gh/SeldonIO/alibi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alibi?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/alibi?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/alibi?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![GitHub - License](https://img.shields.io/github/license/SeldonIO/alibi?logo=github&style=flat&color=green)][#github-license]
[![Slack channel](https://img.shields.io/badge/chat-on%20slack-e51670.svg)][#slack-channel]

<!--- Hide platform for now as platform agnostic --->
<!--- [![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/alibi?logo=anaconda&style=flat)][#conda-forge-package]--->

[#github-license]: https://github.com/SeldonIO/alibi/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/alibi/
[#conda-forge-package]: https://anaconda.org/conda-forge/alibi
[#docs-package]: https://docs.seldon.io/projects/alibi/en/stable/
[#build-status]: https://github.com/SeldonIO/alibi/actions?query=workflow%3A%22CI%22
[#slack-channel]: https://join.slack.com/t/seldondev/shared_invite/zt-vejg6ttd-ksZiQs3O_HOtPQsen_labg
<!--- BADGES: END --->
---

[Alibi](https://docs.seldon.io/projects/alibi) is an open source Python library aimed at machine learning model inspection and interpretation.
The focus of the library is to provide high-quality implementations of black-box, white-box, local and global
explanation methods for classification and regression models.
*  [Documentation](https://docs.seldon.io/projects/alibi/en/stable/)

If you're interested in outlier detection, concept drift or adversarial instance detection, check out our sister project [alibi-detect](https://github.com/SeldonIO/alibi-detect).

<table>
  <tr valign="top">
    <td width="50%" >
        <a href="https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html">
            <br>
            <b>Anchor explanations for images</b>
            <br>
            <br>
            <img src="https://github.com/SeldonIO/alibi/raw/master/doc/source/_static/anchor_image.png">
        </a>
    </td>
    <td width="50%">
        <a href="https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_imdb.html">
            <br>
            <b>Integrated Gradients for text</b>
            <br>
            <br>
            <img src="https://github.com/SeldonIO/alibi/raw/master/doc/source/_static/ig_text.png">
        </a>
    </td>
  </tr>
  <tr valign="top">
    <td width="50%">
        <a href="https://docs.seldon.io/projects/alibi/en/stable/methods/CFProto.html">
            <br>
            <b>Counterfactual examples</b>
            <br>
            <br>
            <img src="https://github.com/SeldonIO/alibi/raw/master/doc/source/_static/cf.png">
        </a>
    </td>
    <td width="50%">
        <a href="https://docs.seldon.io/projects/alibi/en/stable/methods/ALE.html">
            <br>
            <b>Accumulated Local Effects</b>
            <br>
            <br>
            <img src="https://github.com/SeldonIO/alibi/raw/master/doc/source/_static/ale.png">
        </a>
    </td>
  </tr>
</table>

## Table of Contents

* [Installation and Usage](#installation-and-usage)
* [Supported Methods](#supported-methods)
  * [Model Explanations](#model-explanations)
  * [Model Confidence](#model-confidence)
  * [Prototypes](#prototypes)
  * [References and Examples](#references-and-examples)
* [Citations](#citations)

## Installation and Usage
Alibi can be installed from:

- PyPI or GitHub source (with `pip`)
- Anaconda (with `conda`/`mamba`)

### With pip

- Alibi can be installed from [PyPI](https://pypi.org/project/alibi):

  ```bash
  pip install alibi
  ```
  
- Alternatively, the development version can be installed:
  ```bash
  pip install git+https://github.com/SeldonIO/alibi.git 
  ```

- To take advantage of distributed computation of explanations, install `alibi` with `ray`:
  ```bash
  pip install alibi[ray]
  ```

- For SHAP support, install `alibi` as follows:
  ```bash
  pip install alibi[shap]
  ```

### With conda 

To install from [conda-forge](https://conda-forge.org/) it is recommended to use [mamba](https://mamba.readthedocs.io/en/stable/), 
which can be installed to the *base* conda enviroment with:

```bash
conda install mamba -n base -c conda-forge
```

- For the standard Alibi install:
  ```bash
  mamba install -c conda-forge alibi
  ```

- For distributed computing support:
  ```bash
  mamba install -c conda-forge alibi ray
  ```

- For SHAP support:
  ```bash
  mamba install -c conda-forge alibi shap
  ```

### Usage
The alibi explanation API takes inspiration from `scikit-learn`, consisting of distinct initialize,
fit and explain steps. We will use the [AnchorTabular](https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html)
explainer to illustrate the API:

```python
from alibi.explainers import AnchorTabular

# initialize and fit explainer by passing a prediction function and any other required arguments
explainer = AnchorTabular(predict_fn, feature_names=feature_names, category_map=category_map)
explainer.fit(X_train)

# explain an instance
explanation = explainer.explain(x)
```

The explanation returned is an `Explanation` object with attributes `meta` and `data`. `meta` is a dictionary
containing the explainer metadata and any hyperparameters and `data` is a dictionary containing everything
related to the computed explanation. For example, for the Anchor algorithm the explanation can be accessed
via `explanation.data['anchor']` (or `explanation.anchor`). The exact details of available fields varies
from method to method so we encourage the reader to become familiar with the
[types of methods supported](https://docs.seldon.io/projects/alibi/en/stable/overview/algorithms.html).
 

## Supported Methods
The following tables summarize the possible use cases for each method.

### Model Explanations
| Method                                                                                                       |    Models    |     Explanations      | Classification | Regression | Tabular | Text | Images | Categorical features | Train set required | Distributed |
|:-------------------------------------------------------------------------------------------------------------|:------------:|:---------------------:|:--------------:|:----------:|:-------:|:----:|:------:|:--------------------:|:------------------:|:-----------:|
| [ALE](https://docs.seldon.io/projects/alibi/en/stable/methods/ALE.html)                                      |      BB      |        global         |       ✔        |     ✔      |    ✔    |      |        |                      |                    |             |
| [Partial Dependence](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html)         |    BB WB     |        global         |       ✔        |     ✔      |    ✔    |      |        |          ✔           |                    |             |
| [PD Variance](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html)        |    BB WB     |        global         |       ✔        |     ✔      |    ✔    |      |        |          ✔           |                    |             |
| [Permutation Importance](https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html) |      BB      |        global         |       ✔        |     ✔      |    ✔    |      |        |          ✔           |                    |             |
| [Anchors](https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html)                              |      BB      |         local         |       ✔        |            |    ✔    |  ✔   |   ✔    |          ✔           |    For Tabular     |             |
| [CEM](https://docs.seldon.io/projects/alibi/en/stable/methods/CEM.html)                                      | BB* TF/Keras |         local         |       ✔        |            |    ✔    |      |   ✔    |                      |      Optional      |             |
| [Counterfactuals](https://docs.seldon.io/projects/alibi/en/stable/methods/CF.html)                           | BB* TF/Keras |         local         |       ✔        |            |    ✔    |      |   ✔    |                      |         No         |             |
| [Prototype Counterfactuals](https://docs.seldon.io/projects/alibi/en/stable/methods/CFProto.html)            | BB* TF/Keras |         local         |       ✔        |            |    ✔    |      |   ✔    |          ✔           |      Optional      |             |
| [Counterfactuals with RL](https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html)                 |      BB      |         local         |       ✔        |            |    ✔    |      |   ✔    |          ✔           |         ✔          |             |
| [Integrated Gradients](https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html)     |   TF/Keras   |         local         |       ✔        |     ✔      |    ✔    |  ✔   |   ✔    |          ✔           |      Optional      |             |
| [Kernel SHAP](https://docs.seldon.io/projects/alibi/en/stable/methods/KernelSHAP.html)                       |      BB      | local <br></br>global |       ✔        |     ✔      |    ✔    |      |        |          ✔           |         ✔          |      ✔      |
| [Tree SHAP](https://docs.seldon.io/projects/alibi/en/stable/methods/TreeSHAP.html)                           |      WB      | local <br></br>global |       ✔        |     ✔      |    ✔    |      |        |          ✔           |      Optional      |             |
| [Similarity explanations](https://docs.seldon.io/projects/alibi/en/stable/methods/Similarity.html)           |      WB      |         local         |       ✔        |     ✔      |    ✔    |  ✔   |   ✔    |          ✔           |         ✔          |             |

### Model Confidence
These algorithms provide **instance-specific** scores measuring the model confidence for making a
particular prediction.

|Method|Models|Classification|Regression|Tabular|Text|Images|Categorical Features|Train set required|
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
|[Trust Scores](https://docs.seldon.io/projects/alibi/en/stable/methods/TrustScores.html)|BB|✔| |✔|✔(1)|✔(2)| |Yes|
|[Linearity Measure](https://docs.seldon.io/projects/alibi/en/stable/methods/LinearityMeasure.html)|BB|✔|✔|✔| |✔| |Optional|

Key:
 - **BB** - black-box (only require a prediction function)
 - **BB\*** - black-box but assume model is differentiable
 - **WB** - requires white-box model access. There may be limitations on models supported
 - **TF/Keras** - TensorFlow models via the Keras API
 - **Local** - instance specific explanation, why was this prediction made?
 - **Global** - explains the model with respect to a set of instances
 - **(1)** -  depending on model
 - **(2)** -  may require dimensionality reduction

### Prototypes
These algorithms provide a **distilled** view of the dataset and help construct a 1-KNN **interpretable** classifier.

|Method|Classification|Regression|Tabular|Text|Images|Categorical Features|Train set labels|
|:-----|:-------------|:---------|:------|:---|:-----|:-------------------|:---------------|
|[ProtoSelect](https://docs.seldon.io/projects/alibi/en/latest/methods/ProtoSelect.html)|✔| |✔|✔|✔|✔| Optional       |


## References and Examples
- Accumulated Local Effects (ALE, [Apley and Zhu, 2016](https://arxiv.org/abs/1612.08468))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/ALE.html)
  - Examples:
    [California housing dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/ale_regression_california.html),
    [Iris dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/ale_classification.html)

- Partial Dependence ([J.H. Friedman, 2001](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boostingmachine/10.1214/aos/1013203451.full))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html)
  - Examples:
    [Bike rental](https://docs.seldon.io/projects/alibi/en/stable/examples/pdp_regression_bike.html)

- Partial Dependence Variance([Greenwell et al., 2018](https://arxiv.org/abs/1805.04755))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html)
  - Examples:
    [Friedman’s regression problem](https://docs.seldon.io/projects/alibi/en/stable/examples/pd_variance_regression_friedman.html)

- Permutation Importance([Breiman, 2001](https://link.springer.com/article/10.1023/A:1010933404324); [Fisher et al., 2018](https://arxiv.org/abs/1801.01489))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html)
  - Examples:
    [Who's Going to Leave Next?](https://docs.seldon.io/projects/alibi/en/stable/examples/permutation_importance_classification_leave.html)

- Anchor explanations ([Ribeiro et al., 2018](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html)
  - Examples:
    [income prediction](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_tabular_adult.html),
    [Iris dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_tabular_iris.html),
    [movie sentiment classification](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_text_movie.html),
    [ImageNet](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html),
    [fashion MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_fashion_mnist.html)

- Contrastive Explanation Method (CEM, [Dhurandhar et al., 2018](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/CEM.html)
  - Examples: [MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/cem_mnist.html),
    [Iris dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/cem_iris.html)

- Counterfactual Explanations (extension of
  [Wachter et al., 2017](https://arxiv.org/abs/1711.00399))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/CF.html)
  - Examples: 
    [MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/cf_mnist.html)

- Counterfactual Explanations Guided by Prototypes ([Van Looveren and Klaise, 2019](https://arxiv.org/abs/1907.02584))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/CFProto.html)
  - Examples:
    [MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/cfproto_mnist.html),
    [California housing dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/cfproto_housing.html),
    [Adult income (one-hot)](https://docs.seldon.io/projects/alibi/en/stable/examples/cfproto_cat_adult_ohe.html),
    [Adult income (ordinal)](https://docs.seldon.io/projects/alibi/en/stable/examples/cfproto_cat_adult_ord.html)

- Model-agnostic Counterfactual Explanations via RL([Samoilescu et al., 2021](https://arxiv.org/abs/2106.02597))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html)
  - Examples:
    [MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/cfrl_mnist.html),
    [Adult income](https://docs.seldon.io/projects/alibi/en/stable/examples/cfrl_adult.html)

- Integrated Gradients ([Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html),
  - Examples:
    [MNIST example](https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_mnist.html),
    [Imagenet example](https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_imagenet.html),
    [IMDB example](https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_imdb.html).

- Kernel Shapley Additive Explanations ([Lundberg et al., 2017](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/KernelSHAP.html)
  - Examples:
    [SVM with continuous data](https://docs.seldon.io/projects/alibi/en/stable/examples/kernel_shap_wine_intro.html),
    [multinomial logistic regression with continous data](https://docs.seldon.io/projects/alibi/en/stable/examples/kernel_shap_wine_lr.html),
    [handling categorical variables](https://docs.seldon.io/projects/alibi/en/stable/examples/kernel_shap_adult_lr.html)
    
- Tree Shapley Additive Explanations ([Lundberg et al., 2020](https://www.nature.com/articles/s42256-019-0138-9))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/TreeSHAP.html)
  - Examples:
    [Interventional (adult income, xgboost)](https://docs.seldon.io/projects/alibi/en/stable/examples/interventional_tree_shap_adult_xgb.html),
    [Path-dependent (adult income, xgboost)](https://docs.seldon.io/projects/alibi/en/stable/examples/path_dependent_tree_shap_adult_xgb.html)
    
- Trust Scores ([Jiang et al., 2018](https://arxiv.org/abs/1805.11783))
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/TrustScores.html)
  - Examples:
    [MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/trustscore_mnist.html),
    [Iris dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/trustscore_mnist.html)

- Linearity Measure
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/LinearityMeasure.html)
  - Examples:
    [Iris dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/linearity_measure_iris.html),
    [fashion MNIST](https://docs.seldon.io/projects/alibi/en/stable/examples/linearity_measure_fashion_mnist.html)

- ProtoSelect
  - [Documentation](https://docs.seldon.io/projects/alibi/en/latest/methods/ProtoSelect.html)
  - Examples:
    [Adult Census & CIFAR10](https://docs.seldon.io/projects/alibi/en/latest/examples/protoselect_adult_cifar10.html)

- Similarity explanations
  - [Documentation](https://docs.seldon.io/projects/alibi/en/stable/methods/Similarity.html)
  - Examples:
    [20 news groups dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/similarity_explanations_20ng.html),
    [ImageNet dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/similarity_explanations_imagenet.html),
    [MNIST dataset](https://docs.seldon.io/projects/alibi/en/stable/examples/similarity_explanations_mnist.html)

## Citations
If you use alibi in your research, please consider citing it.

BibTeX entry:

```
@article{JMLR:v22:21-0017,
  author  = {Janis Klaise and Arnaud Van Looveren and Giovanni Vacanti and Alexandru Coca},
  title   = {Alibi Explain: Algorithms for Explaining Machine Learning Models},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {181},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v22/21-0017.html}
}
```

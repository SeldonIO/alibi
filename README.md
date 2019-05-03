<p align="center">
  <img src="doc/source/_static/Alibi_Logo.png" alt="Alibi Logo" width="50%">
</p>

[Alibi](https://docs.seldon.io/projects/alibi) is an open source Python library aimed at machine learning model inspection and interpretation. The initial focus on the library is on black-box, instance based model explanations.

*  [Documentation](https://docs.seldon.io/projects/alibi)

## Goals
* Provide high quality reference implementations of black-box ML model explanation algorithms
* Define a consistent API for interpretable ML methods
* Support multiple use cases (e.g. tabular, text and image data classification, regression)
* Implement the latest model explanation, concept drift, algorithmic bias detection and other ML
  model monitoring and interpretation methods

## Installation
Alibi can be installed from [PyPI](https://pypi.org/project/alibi):
```bash
pip install alibi
```

## Examples

[**Anchor method applied to the InceptionV3 model trained on ImageNet:**](examples/anchor_image_imagenet.ipynb)

Prediction: Persian Cat             | Anchor explanation
:-------------------------:|:------------------:
![Persian Cat](doc/source/methods/persiancat.png)| ![Persian Cat Anchor](doc/source/methods/persiancatanchor.png)

[**Contrastive Explanation method applied to a CNN trained on MNIST:**](examples/cem_mnist.ipynb)

Prediction: 4             |  Pertinent Negative: 9               | Pertinent Positive: 4
:-------------------------:|:-------------------:|:------------------:
![mnist_orig](doc/source/methods/mnist_orig.png)  | ![mnsit_pn](doc/source/methods/mnist_pn.png) | ![mnist_pp](doc/source/methods/mnist_pp.png)

[**Trust scores applied to a softmax classifier trained on MNIST:**](examples/trustscore_mnist.ipynb)

![trust_mnist](doc/source/_static/trustscores.png)


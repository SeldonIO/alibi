# Roadmap
Alibi aims to be the go-to library for ML model interpretability. There are multiple
challenges for developing a high-quality, production-ready library that achieves this. In addition
to having high quality reference implementations of the most promising algorithms, we need extensive
documentation and case studies comparing the different interpretability methods and their respective
pros and cons. A clean and a usable API is also a priority.

## Short term
* White-box explanation methods (e.g. TreeSHAP, Integrated Gradients)
* Support both TensorFlow and PyTorch for white-box methods

## Medium term
* Enable TensorFlow 2.0 support for white-box and counterfactual methods ([Github issue](https://github.com/SeldonIO/alibi/issues/155))

## Long term
* Ongoing optimizations of existing algorithms (speed, parallelisation, explanation quality)
* Explanations for sequential and structured data

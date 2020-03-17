# Roadmap
Alibi aims to be the go-to library for ML model interpretability and monitoring. There are multiple
challenges for developing a high-quality, production-ready library that achieves this. In addition
to having high quality reference implementations of the most promising algorithms, we need extensive
documentation and case studies comparing the different interpretability methods and their respective
pros and cons. A clean and a usable API is also a priority.

## Short term
* Ongoing optimizations of existing algorithms (speed, parallelisation, explanation quality)
* Finalize a unified API ([Github PR](https://github.com/SeldonIO/alibi/pull/166))
* Initial visualizations and visualization backends ([Github issue](https://github.com/SeldonIO/alibi/issues/165))
* White-box explanation methods (e.g. Integrated Gradients)
* Support both TensorFlow and PyTorch for white-box methods

## Medium term
* Migrate counterfactual methods to TensorFlow 2.0 ([Github issue](https://github.com/SeldonIO/alibi/issues/155))
* Additional black-box explanation methods ([ALE](https://github.com/SeldonIO/alibi/pull/152))
* Additional model confidence/calibration methods

## Long term
* Explanations for regression models
* Explanations for sequential and structured data

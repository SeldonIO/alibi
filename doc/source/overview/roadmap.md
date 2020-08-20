# Roadmap
Alibi aims to be the go-to library for ML model interpretability. There are multiple
challenges for developing a high quality, production-ready library that achieves this. In addition
to having high quality reference implementations of the most promising algorithms, we need extensive
documentation and case studies comparing the different interpretability methods and their respective
pros and cons. A clean and a usable API is also a priority.

## Short term
* Complete refactoring to enable multiple backends (TensorFlow, PyTorch) and distributed computing
* AnchorText improvements using generative models

## Medium term
* PyTorch support for white-box gradient based explanations
* Improve black-box counterfactual explanations using gradient-free methods

## Long term
* Ongoing optimizations of existing algorithms (speed, parallelisation, explanation quality)
* Explanations for sequential and structured data

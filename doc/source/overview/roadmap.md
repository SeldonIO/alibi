# Roadmap
Alibi aims to be the go-to library for ML model interpretability and monitoring. There are multiple
challenges for developing a high-quality, production-ready library that achieves this. In addition
to having high quality reference implementations of the most promising algorithms, we need extensive
documentation and case studies comparing the different interpretability methods and their respective
pros and cons. A clean and a usable API is also a priority. Additionally we want to move beyond
model explanation and provide tools to gauge ML model confidence, measure concept drift, detect
outliers and bias among other things.

## Additional explanation methods
* [Counterfactual examples](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
  [[WIP](https://github.com/SeldonIO/alibi/pull/35)]
* [Influence functions](https://arxiv.org/abs/1703.04730)
* Feature attribution methods (e.g. [SHAP](https://github.com/slundberg/shap))
* Global methods (e.g. [ALE](https://christophm.github.io/interpretable-ml-book/ale.html#fn31))

## Important enhancements to explanation methods
* Robust handling of categorical variables
  ([Github issue](https://github.com/SeldonIO/alibi/issues/33))
* Document pitfalls of popular methods like LIME and PDP
  ([Github issue](https://github.com/SeldonIO/alibi/issues/42))
* Unified API ([Github issue](https://github.com/SeldonIO/alibi/issues/23))
* Standardized return types for explanations
* Explanations for regression models ([Github issue](https://github.com/SeldonIO/alibi/issues/19))
* Explanations for sequential data
* Develop methods for highly correlated features

## Beyond explanations
* Investigate alternatives to Trust Scores for gauging the confidence of black-box models
* Concept drift - provide methods for monitoring and alerting to changes in the incoming data
  distribution and the conditional distribution of the predictions
* Bias detection methods
* Outlier detection methods ([Github issue](https://github.com/SeldonIO/alibi/issues/13))
# Table of contents

## Overview

* [Introduction](source/overview/high_level.md)
* [Getting Started](source/overview/getting_started.md)
* [Algorithm Overview](source/overview/algorithms.md)
* [White-box and black-box models](source/overview/white_box_black_box.md)
* [Saving and loading](source/overview/saving.md)
* [Frequently Asked Questions](source/overview/faq.md)

## Explanations
* [Methods](source/methods/README.md)
  * [ALE](source/methods/ale.md)
  * [Anchors](source/methods/anchors.md)
  * [CEM](source/methods/cem.md)
  * [CF](source/methods/cf.md)
  * [CFProto](source/methods/cfproto.md)
  * [CFRL](source/methods/cfrl.md)
  * [IntegratedGradients](source/methods/integratedgradients.md)
  * [KernelSHAP](source/methods/kernelshap.md)
  * [LinearityMeasure](source/methods/linearitymeasure.md)
  * [PartialDependence](source/methods/partialdependence.md)
  * [PartialDependenceVariance](source/methods/partialdependencevariance.md)
  * [PermutationImportance](source/methods/permutationimportance.md)
  * [ProtoSelect](source/methods/protoselect.md)
  * [Similarity](source/methods/similarity.md)
  * [TreeSHAP](source/methods/treeshap.md)
  * [TrustScores](source/methods/trustscores.md)
* [Examples](source/methods/README.md)
  * [Alibi Overview Examples](source/examples/overview.md)
  * [Accumulated Local Effets]
    * [Accumulated Local Effects for classifying flowers](source/examples/ale_classification.md)
    * [Accumulated Local Effects for predicting house prices](source/examples/ale_regression_california.md)
  * [Anchors]
    * [Anchor explanations for fashion MNIST](source/examples/anchor_image_fashion_mnist.md)
    * [Anchor explanations for ImageNet](source/examples/anchor_image_imagenet.md)
    * [Anchor explanations for income prediction](source/examples/anchor_tabular_adult.md)
    * [Anchor explanations on the Iris dataset](source/examples/anchor_tabular_iris.md)
    * [Anchor explanations for movie sentiment](source/examples/anchor_text_movie.md)
  * [Contrastive Explanation Method]
    * [Contrastive Explanations Method (CEM) applied to Iris dataset](source/examples/cem_iris.md)
    * [Contrastive Explanations Method (CEM) applied to MNIST](source/examples/cem_mnist.md)
  * [Counterfactual Instances on MNIST](source/examples/cf_mnist.md)
  * [Counterfactuals Guided by Prototypes]
    * [Counterfactual explanations with one-hot encoded categorical variables](source/examples/cfproto_cat_adult_ohe.md)
    * [Counterfactual explanations with ordinally encoded categorical variables](source/examples/cfproto_cat_adult_ord.md)
    * [Counterfactuals guided by prototypes on California housing dataset](source/examples/cfproto_housing.md)
    * [Counterfactuals guided by prototypes on MNIST](source/examples/cfproto_mnist.md)
  * [Counterfactuals with Reinforcement Learning]
    * [Counterfactual with Reinforcement Learning (CFRL) on Adult Census](source/examples/cfrl_adult.md)
    * [Counterfactual with Reinforcement Learning (CFRL) on MNIST](source/examples/cfrl_mnist.md)
  * [Integrated Gradients]
    * [Integrated gradients for a ResNet model trained on Imagenet dataset](source/examples/integrated_gradients_imagenet.md)
    * [Integrated gradients for text classification on the IMDB dataset](source/examples/integrated_gradients_imdb.md)
    * [Integrated gradients for MNIST](source/examples/integrated_gradients_mnist.md)
    * [Integrated gradients for transformers models](source/examples/integrated_gradients_transformers.md)
  * [Kernel SHAP]
    * [Distributed KernelSHAP](source/examples/distributed_kernel_shap_adult_lr.md)
    * [KernelSHAP: combining preprocessor and predictor](source/examples/kernel_shap_adult_categorical_preproc.md)
    * [Handling categorical variables with KernelSHAP](source/examples/kernel_shap_adult_lr.md)
    * [Kernel SHAP explanation for SVM models](source/examples/kernel_shap_wine_intro.md)
    * [Kernel SHAP explanation for multinomial logistic regression models](source/examples/kernel_shap_wine_lr.md)
  * [Partial Dependence]
    * [Partial Dependence and Individual Conditional Expectation for predicting bike renting](source/examples/pdp_regression_bike.md)
  * [Partial Dependence Variance]
    * [Feature importance and feature interaction based on partial dependece variance](source/examples/pd_variance_regression_friedman.md)
  * [Permutation Importance]
    * [Permutation Feature Importance on “Who’s Going to Leave Next?”](source/examples/permutation_importance_classification_leave.md)
  * [Similarity explanations]
    * [Similarity explanations for 20 newsgroups dataset](source/examples/similarity_explanations_20ng.md)
    * [Similarity explanations for ImageNet](source/examples/similarity_explanations_imagenet.md)
    * [Similarity explanations for MNIST](source/examples/similarity_explanations_mnist.md)
  * [Tree SHAP]
    * [Explaining Tree Models with Interventional Feature Perturbation Tree SHAP](source/examples/interventional_tree_shap_adult_xgb.md)
    * [Explaining Tree Models with Path-Dependent Feature Perturbation Tree SHAP](source/examples/path_dependent_tree_shap_adult_xgb.md)

## Model Confidence

* [Methods]
  * [Measuring the linearity of machine learning models](source/methods/linearitymeasure.md)
  * [Trust Scores](source/methods/trustscores.md)
* [Examples]
  * [Measuring the linearity of machine learning models]
    * [Linearity measure applied to fashion MNIST](source/examples/linearity_measure_fashion_mnist.md)
    * [Linearity measure applied to Iris](source/examples/linearity_measure_iris.md)
  * [Trust Scores]
    * [Trust Scores applied to Iris](source/examples/trustscore_iris.md)
    * [Trust Scores applied to MNIST](source/examples/trustscore_mnist.md)

## Prototypes

* [prototypes](source/prototypes/README.md)
  * [Methods](source/prototypes/methods.md)
  * [Examples](source/prototypes/examples.md)
  

## API Reference

* [\[Annotation Based Configuration\]](api-reference/annotation-based-configuration.md)

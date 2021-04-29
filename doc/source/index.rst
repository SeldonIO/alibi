.. alibi documentation master file, created by
   sphinx-quickstart on Thu Feb 28 11:04:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. mdinclude:: landing.md

.. toctree::
  :maxdepth: 1
  :caption: Overview

  overview/getting_started
  overview/algorithms
  overview/saving
  overview/roadmap

.. toctree::
   :maxdepth: 1
   :caption: Methods

   methods/ALE.ipynb
   methods/Anchors.ipynb
   methods/CEM.ipynb
   methods/CF.ipynb
   methods/CFProto.ipynb
   methods/IntegratedGradients.ipynb
   methods/KernelSHAP.ipynb
   methods/LinearityMeasure.ipynb
   methods/TrustScores.ipynb
   methods/TreeSHAP.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/ale_regression_boston
   examples/ale_classification
   examples/anchor_tabular_adult
   examples/anchor_tabular_iris
   examples/anchor_text_movie
   examples/anchor_image_imagenet
   examples/anchor_image_fashion_mnist
   examples/cem_mnist
   examples/cem_iris
   examples/cf_mnist.ipynb
   examples/cfproto_mnist.ipynb
   examples/cfproto_housing.ipynb
   examples/cfproto_cat_adult_ohe.ipynb
   examples/cfproto_cat_adult_ord.ipynb
   examples/kernel_shap_wine_intro
   examples/kernel_shap_wine_lr
   examples/kernel_shap_adult_lr
   examples/kernel_shap_adult_categorical_preproc
   examples/distributed_kernel_shap_adult_lr
   examples/linearity_measure_iris
   examples/linearity_measure_fashion_mnist
   examples/trustscore_iris
   examples/trustscore_mnist
   examples/interventional_tree_shap_adult_xgb
   examples/path_dependent_tree_shap_adult_xgb
   examples/integrated_gradients_imagenet.ipynb
   examples/integrated_gradients_mnist.ipynb
   examples/integrated_gradients_imdb.ipynb

.. toctree::
   :maxdepth: 1
   :caption: API reference

   API reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

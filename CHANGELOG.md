# Change Log

## [v0.9.0](https://github.com/SeldonIO/alibi/tree/v0.9.0) (2023-01-11)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.8.0...v0.9.0)

### Added
- **New feature** `PermutationImportance` explainer implementing the permutation feature importance global explanations. Also included is a `plot_permutation_importance` utility function for flexible plotting of the resulting feature importance scores ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html),  [#798](https://github.com/SeldonIO/alibi/pull/798)). 
- **New feature** `PartialDependenceVariance` explainer implementing partial dependence variance global explanations. Also included is a `plot_pd_variance` utility function for flexible plotting of the resulting PD variance plots ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependenceVariance.html), [#758](https://github.com/SeldonIO/alibi/pull/758)).

### Fixed
- `GradientSimilarity` explainer now automatically handles sparse tensors in the model by converting the gradient tensors to dense ones before calculating similarity. This used to be a source of bugs when calculating similarity for models with embedding layers for which gradients tensors are sparse by default. Additionally, it now filters any non-trainable parameters and doesn't consider those in the calculation as no gradients exist. A warning is raised if any non-trainable layers or parameters are detected ([#829](https://github.com/SeldonIO/alibi/pull/829)).
- Updated the discussion of the interpretation of `ALE`. The previous examples and documentation had some misleading claims; these have been removed and reworked with an emphasis on the mostly qualitative interpretation of `ALE` plots ([#838](https://github.com/SeldonIO/alibi/pull/838), [#846](https://github.com/SeldonIO/alibi/pull/846)).

### Changed
- Deprecated the use of the legacy Boston housing dataset in examples and testing. The new examples now use the California housing dataset ([#838](https://github.com/SeldonIO/alibi/pull/838), [#834](https://github.com/SeldonIO/alibi/pull/834)).
- Modularized the computation of prototype importances and plotting for `ProtoSelect`, allowing greater flexibility to the end user ([#826](https://github.com/SeldonIO/alibi/pull/826)).
- Roadmap documentation page removed due to going out of date ([#842](https://github.com/SeldonIO/alibi/pull/842)).

### Development
- Tests added for `tensorflow` models used in `CounterfactualRL` ([#793](https://github.com/SeldonIO/alibi/pull/793)).
- Tests added for `pytorch` models used in `CounterfactualRL` ([#799](https://github.com/SeldonIO/alibi/pull/799)).
- Tests added for `ALE` plotting functionality ([#816](https://github.com/SeldonIO/alibi/pull/816)).
- Tests added for `PartialDependence` plotting functionality ([#819](https://github.com/SeldonIO/alibi/pull/819)).
- Tests added for `PartialDependenceVariance` plotting functionality ([#820](https://github.com/SeldonIO/alibi/pull/820)).
- Tests added for `PermutationImportance` plotting functionality ([#824](https://github.com/SeldonIO/alibi/pull/824)).
- Tests addef for `ProtoSelect` plotting functionality ([#841](https://github.com/SeldonIO/alibi/pull/841)).
- Tests added for the `datasets` subpackage ([#814](https://github.com/SeldonIO/alibi/pull/814)).
- Fixed optional dependency installation during CI to make sure dependencies are consistent ([#817](https://github.com/SeldonIO/alibi/pull/817)).
- Synchronize notebook CI workflow with the main CI workflow ([#818](https://github.com/SeldonIO/alibi/pull/818)).
- Version of `pytest-cov` bumped to `4.x` ([#794](https://github.com/SeldonIO/alibi/pull/794)).
- Version of `pytest-xdist` bumped to `3.x` ([#808](https://github.com/SeldonIO/alibi/pull/808)).
- Version of `tox` bumped to `4.x` ([#832](https://github.com/SeldonIO/alibi/pull/832)).


## [v0.8.0](https://github.com/SeldonIO/alibi/tree/v0.8.0) (2022-09-26)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.7.0...v0.8.0)

### Added
- **New feature** `PartialDependence` and `TreePartialDependence` explainers implementing partial dependence (PD) global explanations. Also included is a `plot_pd` utility function for flexible plotting of the resulting PD plots ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/PartialDependence.html), [#721](https://github.com/SeldonIO/alibi/pull/721)).
- New `exceptions.NotFittedError` exception which is raised whenever a compulsory call to a `fit` method has not been carried out. Specifically, this is now raised in `AnchorTabular.explain` when `AnchorTabular.fit` has been skipped ([#732](https://github.com/SeldonIO/alibi/pull/732)).
- Various improvements to docs and examples ([#695](https://github.com/SeldonIO/alibi/pull/695), [#701](https://github.com/SeldonIO/alibi/pull/701), [#698](https://github.com/SeldonIO/alibi/pull/698), [#703](https://github.com/SeldonIO/alibi/pull/703), [#717](https://github.com/SeldonIO/alibi/pull/717), [#711](https://github.com/SeldonIO/alibi/pull/711), [#750](https://github.com/SeldonIO/alibi/pull/750), [#784](https://github.com/SeldonIO/alibi/pull/784)).

### Fixed
- Edge case in `AnchorTabular` where an error is raised during an `explain` call if the instance contains a categorical feature value not seen in the training data ([#742](https://github.com/SeldonIO/alibi/pull/742)).

### Changed
- Improved handling of custom `grid_points` for the `ALE` explainer ([#731](https://github.com/SeldonIO/alibi/pull/731)).
- Renamed our custom exception classes to remove the verbose `Alibi*` prefix and standardised the `*Error` suffix. Concretely:
  - `exceptions.AlibiPredictorCallException` is now `exceptions.PredictorCallError`
  - `exceptions.AlibiPredictorReturnTypeError` is now `exceptions.PredictorReturnTypeError`. Backwards compatibility has been maintained by subclassing the new exception classes by the old ones, **but these will likely be removed in a future version** ([#733](https://github.com/SeldonIO/alibi/pull/733)).
- Warn users when `TreeShap` is used with more than 100 samples in the background dataset which is due to a limitation in the upstream `shap` package ([#710](https://github.com/SeldonIO/alibi/pull/710)).
- Minimum version of `scikit-learn` bumped to `1.0.0` mainly due to upcoming deprecations ([#776](https://github.com/SeldonIO/alibi/pull/776)).
- Minimum version of `scikit-image` bumped to `0.17.2` to fix a possible bug when using the `slic` segmentation function with `AnchorImage` ([#753](https://github.com/SeldonIO/alibi/pull/753)).
- Maximum supported version of `attrs` bumped to `22.x` ([#727](https://github.com/SeldonIO/alibi/pull/727)).
- Maximum supported version of `tensorflow` bumped to `2.10.x` ([#745](https://github.com/SeldonIO/alibi/pull/745)).
- Maximum supported version of `ray` bumped to `2.x` ([#740](https://github.com/SeldonIO/alibi/pull/740)).
- Maximum supported version of `numba` bumped to `0.56.x` ([#724](https://github.com/SeldonIO/alibi/pull/724)).
- Maximum supported version of `shap` bumped to `0.41.x` ([#702](https://github.com/SeldonIO/alibi/pull/702)).
- Updated `shap` example notebooks to recommend installing `matplotlib==3.5.3` due to failure of `shap` plotting functions with `matplotlib==3.6.0` ([#776](https://github.com/SeldonIO/alibi/pull/776)).

### Development
- Extend optional dependency checks to ensure the correct submodules are present ([#714](https://github.com/SeldonIO/alibi/pull/714)). 
- Introduce `pytest-custom_exit_code` to let notebook CI pass when no notebooks are selected for tests ([#728](https://github.com/SeldonIO/alibi/pull/728)).
- Use UTF-8 encoding when loading `README.md` in `setup.py` to avoid a possible failure of installation for some users ([#744](https://github.com/SeldonIO/alibi/pull/744)).
- Updated guidance for class docstrings ([#743](https://github.com/SeldonIO/alibi/pull/743)).
- Reinstate `ray` tests ([#756](https://github.com/SeldonIO/alibi/pull/756)).
- We now exclude test files from test coverage for a more accurate representation of coverage ([#751](https://github.com/SeldonIO/alibi/pull/751)). Note that this has led to a drop in code covered which will be addressed in due course ([#760](https://github.com/SeldonIO/alibi/issues/760)).
- The Python `3.10.x` version on CI has been pinned to `3.10.6` due to typechecking failures, pending a new release of `mypy` ([#761](https://github.com/SeldonIO/alibi/pull/761)).
- The `test_changed_notebooks` workflow can now be triggered manually and is run on push/PR for any branch ([#762](https://github.com/SeldonIO/alibi/commit/98e962b32c31e7ee670147a44af032b593950b5d)).
- Use `codecov` flags for more granular reporting of code coverage ([#759](https://github.com/SeldonIO/alibi/pull/759)).
- Option to ssh into Github Actions runs for remote debugging of CI pipelines ([#770](https://github.com/SeldonIO/alibi/pull/770)).
- Version of `sphinx` bumped to `5.x` but capped at `<5.1.0` to avoid CI failures ([#722](https://github.com/SeldonIO/alibi/pull/722)).
- Version of `myst-parser` bumped to `0.18.x` ([#693](https://github.com/SeldonIO/alibi/pull/693)).
- Version of `flake8` bumped to `5.x` ([#729](https://github.com/SeldonIO/alibi/pull/729)).
- Version of `ipykernel` bumped to `6.x` ([#431](https://github.com/SeldonIO/alibi/pull/572)).
- Version of `ipython` bumped to `8.x` ([#572](https://github.com/SeldonIO/alibi/pull/572)).
- Version of `pytest` bumped to `7.x` ([#591](https://github.com/SeldonIO/alibi/pull/591)).
- Version of `sphinx-design` bumped to `0.3.0` ([#739](https://github.com/SeldonIO/alibi/pull/739)).
- Version of `nbconvert` bumped to `7.x` ([#738](https://github.com/SeldonIO/alibi/pull/738)).



## [v0.7.0](https://github.com/SeldonIO/alibi/tree/v0.7.0) (2022-05-18)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.5...v0.7.0)

This release introduces two new methods, a `GradientSimilarity` explainer and a `ProtoSelect` data summarisation algorithm.

### Added
- **New feature** `GradientSimilarity` explainer for explaining predictions of gradient-based (PyTorch and TensorFlow) models by returning the most similar training data points from the point of view of the model ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/Similarity.html)).
- **New feature** We have introduced a new subpackage `alibi.prototypes` which contains the `ProtoSelect` algorithm for summarising datasets with a representative set of "prototypes" ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/ProtoSelect.html)).
- `ALE` explainer now can take a custom grid-point per feature to evaluate the `ALE` on. This can help in certain situations when grid-points defined by quantiles might not be the best choice ([docs](https://docs.seldon.io/projects/alibi/en/stable/methods/ALE.html#Usage)).
- Extended the `IntegratedGradients` method target selection to handle explaining any scalar dimension of tensors of any rank (previously only rank-1 and rank-2 were supported). See [#635](https://github.com/SeldonIO/alibi/pull/635).
- Python 3.10 support. Note that `PyTorch` at the time of writing doesn't support Python 3.10 on Windows.

### Fixed
- Fixed a bug which incorrectly handled multi-dimensional scaling in `CounterfactualProto` ([#646](https://github.com/SeldonIO/alibi/pull/646)).
- Fixed a bug in the example using `CounterfactualRLTabular` ([#651](https://github.com/SeldonIO/alibi/pull/651)).

### Changed
- `tensorflow` is now an optional dependency. To use methods that require `tensorflow` you can install `alibi` using `pip install alibi[tensorflow]` which will pull in a supported version. For full instructions for the recommended way of installing optional dependencies please refer to [Installation docs](https://docs.seldon.io/projects/alibi/en/stable/overview/getting_started.html#installation).
- Updated `sklearn` version bounds to `scikit-learn>=0.22.0, <2.0.0`.
- Updated `tensorflow` maximum allowed version to `2.9.x`.

### Development
- This release introduces a way to manage the absence of optional dependencies. In short, the design is such that if an optional dependency is required for an algorithm but missing, at import time the corresponding public (or private in the case of the optional dependency being required for a subset of the functionality of a private class) algorithm class will be replaced by a `MissingDependency` object. For full details on developing `alibi` with optional dependencies see [Contributing: Optional Dependencies](https://github.com/SeldonIO/alibi/blob/master/CONTRIBUTING.md#optional-dependencies).
- The [CONTRIBUTING.md](https://github.com/SeldonIO/alibi/blob/master/CONTRIBUTING.md) has been updated with further instructions for managing optional dependencies (see point above) and more conventions around docstrings.
- We have split the `Explainer` base class into `Base` and `Explainer` to facilitate reusability and better class hierarchy semantics with introducing methods that are not explainers ([#649](https://github.com/SeldonIO/alibi/pull/649)).
- `mypy` has been updated to `~=0.900` which requires additional development dependencies for type stubs, currently only `types-requests` has been necessary to add to `requirements/dev.txt`.
- Fron this release onwards we exclude the directories `doc/` and `examples/` from the source distribution (by adding `prune` directives in `MANIFEST.in`). This results in considerably smaller file sizes for the source distribution.

## [v0.6.5](https://github.com/SeldonIO/alibi/tree/v0.6.5) (2022-03-18)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.4...v0.6.5)

This is a patch release to correct a regression in `CounterfactualProto` introduced in `v0.6.3`.

### Added
- Added a [Frequently Asked Questions](https://docs.seldon.io/projects/alibi/en/stable/overview/faq.html) page to the docs.

### Fixed
- Fix a bug introduced in `v0.6.3` which prevented `CounterfactualProto` working with categorical features ([#612](https://github.com/SeldonIO/alibi/pull/612)).
- Fix an issue with the `LanguageModelSampler` where it would sometimes sample punctuation ([#585](https://github.com/SeldonIO/alibi/pull/585)). 

### Development
- The maximum `tensorflow` version has been bumped from 2.7 to 2.8 ([#588](https://github.com/SeldonIO/alibi/pull/588)).


## [v0.6.4](https://github.com/SeldonIO/alibi/tree/v0.6.4) (2022-02-28)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.3...v0.6.4)

This is a patch release to correct a regression in `AnchorImage` introduced in `v0.6.3`.

### Fixed
- Fix a bug introduced in `v0.6.3` where `AnchorImage` would ignore user `segmentation_kwargs` ([#581](https://github.com/SeldonIO/alibi/pull/581)).

### Development
- The maximum versions of `Pillow` and `scikit-image` have been bumped to 9.x and 0.19.x respectively.

## [v0.6.3](https://github.com/SeldonIO/alibi/tree/v0.6.3) (2022-01-18)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.2...v0.6.3)

### Added
- **New feature** A callback can now be passed to `IntegratedGradients` via the `target_fn` argument, in order to calculate the scalar target dimension from the model output. This is to bypass the requirement of passing `target` directly to `explain` when the `target` of interest may depend on the prediction output. See the example in the [docs](https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html). ([#523](https://github.com/SeldonIO/alibi/pull/523)).
- A new comprehensive [Introduction](https://docs.seldon.io/projects/alibi/en/stable/overview/high_level.html) to explainability added to the documentation ([#510](https://github.com/SeldonIO/alibi/pull/510)).

### Changed
- Python 3.6 has been deprecated from the supported versions as it has reached end-of-life. 

### Fixed
- Fix a bug with passing background images to `AnchorImage` leading to an error ([#542](https://github.com/SeldonIO/alibi/pull/542)).
- Fix a bug with rounding errors being introduced in `CounterfactualRLTabular` ([#550](https://github.com/SeldonIO/alibi/pull/550)).

### Development
- Docstrings have been updated and consolidated ([#548](https://github.com/SeldonIO/alibi/pull/548)). For developers, docstring conventions have been documented in [CONTRIBUTING.md](https://github.com/SeldonIO/alibi/blob/master/CONTRIBUTING.md#docstrings).
- `numpy` typing has been updated to be compatible with `numpy 1.22` ([#543](https://github.com/SeldonIO/alibi/pull/543)). This is a prerequisite for upgrading to `tensorflow 2.7`. 
- To further improve reliability, strict `Optional` type-checking with `mypy` has been reinstated ([#541](https://github.com/SeldonIO/alibi/pull/541)).
- The Alibi CI tests now include Windows and MacOS platforms ([#575](https://github.com/SeldonIO/alibi/pull/575)).
- The maximum `tensorflow` version has been bumped from 2.6 to 2.7 ([#377](https://github.com/SeldonIO/alibi-detect/pull/377)).


## [v0.6.2](https://github.com/SeldonIO/alibi/tree/v0.6.2) (2021-11-18)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.1...v0.6.2)

### Added
- Documentation on using black-box and white-box models in the context of alibi, [see here](https://docs.seldon.io/projects/alibi/en/stable/overview/white_box_black_box.html).
- `AnchorTabular`, `AnchorImage` and `AnchorText` now expose an additional `dtype` keyword argument with a default value of `np.float32`. This is to ensure that whenever a user `predictor` is called internally with dummy data a correct data type can be ensured ([#506](https://github.com/SeldonIO/alibi/pull/506)).
- Custom exceptions. A new public module `alibi.exceptions` defining the `alibi` exception hierarchy. This introduces two exceptions, `AlibiPredictorCallException` and `AlibiPredictorReturnTypeError`. See [#520](https://github.com/SeldonIO/alibi/pull/520) for more details.

### Changed
- For `AnchorImage`, coerce `image_shape` argument into a tuple to implicitly allow passing a list input which eases use of configuration files. In the future the typing will be improved to be more explicit about allowed types with runtime type checking.
- Updated the minimum `shap` version to the latest `0.40.0` as this fixes an installation issue if `alibi` and `shap` are installed with the same command.

### Fixed
- Fix a bug with version saving being overwritten on subsequent saves ([#481](https://github.com/SeldonIO/alibi/pull/481)).
- Fix a bug in the Integrated Gradients notebook with transformer models due to a regression in the upstream `transformers` library ([#528](https://github.com/SeldonIO/alibi/pull/528)).
- Fix a bug in `IntegratedGradients` with `forward_kwargs` not always being correctly passed ([#525](https://github.com/SeldonIO/alibi/pull/525)).
- Fix a bug resetting `TreeShap` predictor ([#534](https://github.com/SeldonIO/alibi/pull/534)).


### Development
- Now using `readthedocs` Docker image in our CI to replicate the doc building environment exactly. Also enabled `readthedocs` build on PR feature which allows browsing the built docs on every PR.
- New notebook execution testing framework via Github Actions. There are two new GA workflows, [test_all_notebooks](https://github.com/SeldonIO/alibi/actions/workflows/test_all_notebooks.yml) which is run once a week and can be triggered manually, and [test_changed_notebooks](https://github.com/SeldonIO/alibi/actions/workflows/test_changed_notebooks.yml) which detects if any notebooks have been modified in a PR and executes only those. Not all notebooks are amenable to be tested automatically due to long running times or complex software/hardware dependencies. We maintain a list of notebooks to be excluded in the testing script under [testing/test_notebooks.py](testing/test_notebooks.py).
- Now using `myst` (a markdown superset) for more flexible documentation ([#482](https://github.com/SeldonIO/alibi/pull/482)).
- Added a [CITATION.cff](CITATION.cff) file.

## [v0.6.1](https://github.com/SeldonIO/alibi/tree/v0.6.1) (2021-09-02)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.6.0...v0.6.1)

### Added
- **New feature** An implementation of [Model-agnostic and Scalable Counterfactual Explanations via Reinforcement Learning](https://arxiv.org/abs/2106.02597) is now available via `alibi.explainers.CounterfactualRL` and `alibi.explainers.CounterfactualRLTabular` classes. The method is model-agnostic and the implementation is written in both PyTorch and TensorFlow. See [docs](https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html) for more information.

### Changed
- **Future breaking change** The names of `CounterFactual` and `CounterFactualProto` classes have been changed to `Counterfactual` and `CounterfactualProto` respectively for consistency and correctness. The old class names continue working for now but emit a deprecation warning message and will be removed in an upcoming version.
- `dill` behaviour was changed to not extend the `pickle` protocol so that standard usage of `pickle` in a session with `alibi` does not change expected `pickle` behaviour. See [discussion](https://github.com/SeldonIO/alibi/issues/447).
- `AnchorImage` internals refactored to avoid persistent state between `explain` calls.

### Development
- A PR checklist is available under [CONTRIBUTING.md](../CONTRIBUTING.md#pr-checklist). In the future many of these may be turned into automated checks.
- `pandoc` version for docs building updated to `1.19.2` which is what is used on `readthedocs`.
- Citation updated to the JMLR paper.

## [v0.6.0](https://github.com/SeldonIO/alibi/tree/v0.6.0) (2021-07-08)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.8...v0.6.0)

### Added
- **New feature** `AnchorText` now supports sampling according to masked language models via the `transformers` library. See [docs](https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html#id2) and the [example](https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_text_movie.html) for using the new functionality.
- **Breaking change** due to the new masked language model sampling for `AnchorText` the public API for the constructor has changed. See [docs](https://docs.seldon.io/projects/alibi/en/stable/methods/Anchors.html#id2) for a full description of the new API.
- `AnchorTabular` now supports one-hot encoded categorical variables in addition to the default ordinal/label encoded representation of categorical variables.
- `IntegratedGradients` changes to allow explaining a wider variety of models. In particular, a new `forward_kwargs` argument to `explain` allows passing additional arguments to the model and `attribute_to_layer_inputs` flag to allow calculating attributions with respect to layer input instead of output if set to `True`. The API and capabilities now track more closely to the [captum.ai](https://captum.ai/api/) `PyTorch` implementation.
- [Example](https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_transformers.html) of using `IntegratedGradients` to explain `transformer` models.
- Python 3.9 support.

### Fixed
- `IntegratedGradients` - fix the path definition for attributions calculated with respect to an internal layer. Previously the paths were defined in terms of the inputs and baselines, now they are correctly defined in terms of the corresponding layer input/output. 

## [v0.5.8](https://github.com/SeldonIO/alibi/tree/v0.5.8) (2021-04-29)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.7...v0.5.8)

### Added
- Experimental explainer serialization support using `dill`. See [docs](https://docs.seldon.io/projects/alibi/en/stable/overview/saving.html) for more details.

### Fixed
- Handle layers which are not part of `model.layers` for `IntegratedGradients`.

### Development
- Update type hints to be compatible with `numpy` 1.20.
- Separate licence build step in CI, only check licences against latest Python version.

## [v0.5.7](https://github.com/SeldonIO/alibi/tree/v0.5.7) (2021-03-31)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.6...v0.5.7)

### Changed
- Support for `KernelShap` and `TreeShap` now requires installing the `shap` dependency explicitly after installing `alibi`. This can be achieved by running `pip install alibi && pip install alibi[shap]`. The reason for this is that the build process for the upstream `shap` package is not well configured resulting in broken installations as detailed in https://github.com/SeldonIO/alibi/pull/376 and https://github.com/slundberg/shap/pull/1802. We expect this to be a temporary change until changes are made upstream.

### Added
- A `reset_predictor` method for black-box explainers. The intended use case for this is for deploying an already configured explainer to work with a remote predictor endpoint instead of the local predictor used in development.
- `alibi.datasets.load_cats` function which loads a small sample of cat images shipped with the library to be used in examples.

### Fixed
- Deprecated the `alibi.datasets.fetch_imagenet` function as the Imagenet API is no longer available.
- `IntegratedGradients` now works with subclassed TensorFlow models.
- Removed support for calculating attributions wrt multiple layers in `IntegratedGradients` as this was not working properly and is difficult to do in the general case.

### Development
- Fixed an issue with `AnchorTabular` tests not being picked up due to a name change of test data fixtures.

## [v0.5.6](https://github.com/SeldonIO/alibi/tree/v0.5.6) (2021-02-18)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.5...v0.5.6)

### Added
- **Breaking change** `IntegratedGradients` now supports models with multiple inputs. For each input of the model, attributions are calculated and returned in a list. Also extends the method allowing to calculate attributions for multiple internal layers. If a list of layers is passed, a list of attributions is returned. See  https://github.com/SeldonIO/alibi/pull/321.
- `ALE` now supports selecting a subset of features to explain. This can be useful to reduce runtime if only some features are of interest and also indirectly helps dealing with categorical variables by being able to exclude them (as `ALE` does not support categorical variables).

### Fixed
- `AnchorTabular` coverage calculation was incorrect which was caused by incorrectly indexing a list, this is now resolved.
- `ALE` was causing an error when a constant feature was present. This is now handled explicitly and the user has control over how to handle these features. See https://docs.seldon.io/projects/alibi/en/stable/api/alibi.explainers.ale.html#alibi.explainers.ale.ALE for more details.
- Release of Spacy 3.0 broke the `AnchorText` functionality as the way `lexeme_prob` tables are loaded was changed. This is now fixed by explicitly handling the loading depending on the `spacy` version.
- Fixed documentation to refer to the `Explanation` object instead of the old `dict` object.
- Added warning boxes to `CounterFactual`, `CounterFactualProto` and `CEM` docs to explain the necessity of clearing the TensorFlow graph if switching to a new model in the same session.

### Development
- Introduced lower and upper bounds for library and development dependencies to limit the potential for breaking functionality upon new releases of dependencies.
- Added dependabot support to automatically monitor new releases of dependencies (both library and development).
- Switched from Travis CI to Github Actions as the former limited their free tier.
- Removed unused CI provider configs from the repo to reduce clutter.
- Simplified development dependencies to just two files, `requirements/dev.txt` and `requirements/docs.txt`.
- Split out the docs building stage as a separate step on CI as it doesn't need to run on every Python version thus saving time.
- Added `.readthedocs.yml` to control how user-facing docs are built directly from the repo.
- Removed testing related entries to `setup.py` as the workflow is both unused and outdated.
- Avoid `shap==0.38.1` as a dependency as it assumes `IPython` is installed and breaks the installation.

## [v0.5.5](https://github.com/SeldonIO/alibi/tree/v0.5.5) (2020-10-20)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.4...v0.5.5)

### Added
- **New feature** Distributed backend using `ray`. To use, install `ray` using `pip install alibi[ray]`.
- **New feature** `KernelShap` distributed version using the new distributed backend.
- For anchor methods added an explanation field `data['raw']['instances']` which is a batch-wise version of the existing `data['raw']['instance']`. This is in preparation for the eventual batch support for anchor methods.
- Pre-commit hook for `pyupgrade` via `nbqa` for formatting example notebooks using Python 3.6+ syntax.

### Fixed
- Flaky test for distributed anchors (note: this is the old non-batchwise implementation) by dropping the precision treshold.
- Notebook string formatting upgraded to Python 3.6+ f-strings.

### Changed
- **Breaking change** For anchor methods, the returned explanation field `data['raw']['prediction']` is now batch-wise, i.e. for `AnchorTabular` and `AnchorImage` it is a 1-dimensional `numpy` array whilst for `AnchorText` it is a list of strings. This is in preparation for the eventual batch support for anchor methods.
- Removed dependency on `prettyprinter` and substituted with a slightly modified standard library version of `PrettyPrinter`. This is to prepare for a `conda` release which requires all dependencies to also be published on `conda`.

## [v0.5.4](https://github.com/SeldonIO/alibi/tree/v0.5.4) (2020-09-03)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.3...v0.5.4)

### Added
- `update_metadata` method for any `Explainer` object to enable easy book-keeping for algorithm parameters

### Fixed
- Updated `KernelShap` wrapper to work with the newest `shap>=0.36` library
- Fix some missing metadata parameters in `KernelShap` and `TreeShap`

## [v0.5.3](https://github.com/SeldonIO/alibi/tree/v0.5.3) (2020-09-01)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.2...v0.5.3)

### Changed
- Updated roadmap 

### Fixed
- Bug in integrated gradients where incorrect layer handling led to output shape mismatch when explaining layer outputs
- Remove tf.logging calls in example notebooks as TF 2.x API no longer supports tf.logging
- Pin shap to 0.35.0, pending shap 0.36.0 patch release to support shap API updates/library refactoring

## [v0.5.2](https://github.com/SeldonIO/alibi/tree/v0.5.2) (2020-08-05)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.1...v0.5.2)

This release changes the required TensorFlow version from <2.0 to >=2.0. This means that `alibi` code depends on TenorFlow>=2.0, however the explainer algorithms are compatible for models trained with both TF1.x and TF2.x.

The `alibi` code that depends on TensorFlow itself has not been fully migrated in the sense that the code is still not idiomatic TF2.x code just that we now use the `tf.compat.v1` package provided by TF2.x internally. This does mean that for the time being to run algorithms which depend on TensorFlow (`CounterFactual`, `CEM` and `CounterFactualProto`) require disabling TF2.x behaviour by running `tf.compat.v1.disable_v2_behavior()`. This is documented in the example notebooks. Work is underway to re-write the TensorFlow dependent components in idiomatic TF2.x code so that this will not be necessary in a future release.

The upgrade to TensorFlow 2.x also enables this to be the first release with Python 3.8 support.

Finally, white-box explainers are now tested with pre-trained models from both TF1.x and TF2.x. The binaries for the models along with loading functionality and datasets used to train these are available in the `alibi-testing` helper package which is now a requirement for running tests.

### Changed
- Minimum required TensorFlow version is now 2.0
- Tests depending on trained models are now run using pre-trained models hosted under the `alibi-testing` helper package

### Fixed
- A bug in `AnchorText` resulting from missing string hash entries in some spacy models (https://github.com/SeldonIO/alibi/pull/276)
- Explicitly import `lazy_fixture` in tests instead of relying on the deprecated usage of `pytest` namespace (https://github.com/SeldonIO/alibi/pull/281)
- A few bugs in example notebooks

## [v0.5.1](https://github.com/SeldonIO/alibi/tree/v0.5.1) (2020-07-10)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.0...v0.5.1)

This is a bug fix release.

### Fixed
- Fix an issue with `AnchorText` not working on text instances with commas due to not checking for empty synonym lists
- Enable correct behaviour of `AnchorText` with `spacy>=2.3.0`, this now requires installing `spacy[lookups]` as an additional dependency which contains model probability tables
- Update the `expected_value` attribute of `TreeSHAP` which is internally updated after a call to `explain`
- Fix some links in Integrated Gradients examples
- Coverage after running tests on Travis is now correctly reported as the reports are merged for different `pytest` runs
- Old `Keras` tests now require `Keras<2.4.0` as the new release requires `tensorflow>=2.2`
- Bump `typing_extensions>=3.7.2` which includes the type `Literal`

## [v0.5.0](https://github.com/SeldonIO/alibi/tree/v0.5.0) (2020-06-10)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.4.0...v0.5.0)

This version supports Python 3.6 and 3.7 as support for Python 3.5 is dropped.

### Added
- **New feature** `TreeSHAP` explainer for white-box, tree based model SHAP value computation
- **New feature** `ALE` explainer for computing feature effects for black-box, tabular data models
- **New feature** `IntegratedGradients` explainer for computing feature attributions for TensorFlow and Keras models
- Experimental `utils.visualization` module currently containing visualization functions for `IntegratedGradients` on image datasets.The location, implementation and content of the module and functions therein are subject to change.
- Extend `datasets.fetch_imagenet` to work with any class
- Extend `utils.data.gen_category_map` to take a list of strings of column names

### Changed
- Internal refactoring of `KernelSHAP` to reuse functionality for `TreeSHAP`. Both SHAP wrappers
are now under `explainers.shap_wrappers`
- Tests are now split into two runs, one with TensorFlow in eager mode which is necessary for using `IntegratedGradients`
- Added `typing-extensions` library as a requirement to take advantage of more precise types
- Pinned `scikit-image<0.17` due to a regression upstream
- Pinned `Sphinx<3.0` for documentation builds due to some issues with the `m2r` plugin

### Fixed
- Various improvements to documentation
- Some tests were importing old `keras` functions instead of `tensorflow.keras`


## [v0.4.0](https://github.com/SeldonIO/alibi/tree/v0.4.0) (2020-03-20)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.3.2...v0.4.0)

**NB:** This is the last version supporting Python 3.5.

### Added
- **New feature** `KernelSHAP` explainer for black-box model SHAP scores
- Documentation for the `LinearityMeasure` algorithm
### Changed
- **Breaking change** New API for explainer and explanation objects. Explainer objects now inherit from `Explainer` base class as a minimum. When calling `.explain` method, an `Explanation` object is returned (previously a dictionary). This contains two dictionaries `meta` and `data` accessed as attributes of the object, detailing the metadata and the data of the returned explanation. The common interfaces are under `api.interfaces` and default return metadata and data for each explainer are under `api.defaults`.
- Complete refactoring of the Anchors algorithms, many code improvements
- Explainer tests are now more modular, utilizing scoped fixtures defined in `explainers.tests.conftest` and various utility functions
- Tests are now run sequentially insted of in parallel due to overhead of launching new processes

## [v0.3.2](https://github.com/SeldonIO/alibi/tree/v0.3.2) (2019-10-17)

[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.3.1...v0.3.2)
### Added
- All explanations return a metadata field `meta` with a `name` subfield which is currently the name of the class
### Changed
- Provide URL options for fetching some datasets, by default now fetches from a public Seldon bucket

## [v0.3.1](https://github.com/SeldonIO/alibi/tree/v0.3.1) (2019-10-01)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.3.0...v0.3.1)
### Fixed
- Pin `tensorflow` dependency to versions 1.x as the new 2.0 release introduces breaking changes

## [v0.3.0](https://github.com/SeldonIO/alibi/tree/v0.3.0) (2019-09-25)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.2.3...v0.3.0)
### Added
- **New feature** `LinearityMeasure` class and `linearity_measure` function for measuring the linearity of a classifier/regressor
- **New feature** `CounterFactualProto` now supports categorical variables for tabular data
### Changed
- **Breaking change** Remove need for the user to manage TensorFlow sessions for the explanation methods that use TF internally (`CEM`, `CounterFactual`, `CounterFactualProto`). The session is now inferred or created depending on what is passed to `predict`. For finer control the `sess` parameter can still be passed in directly
- **Breaking change**  Expose low-level arguments to `AnchorText` to the user for finer control of the explanation algorithm, also rename some arguments for consistency
- Various improvements to existing notebook examples
### Fixed
- `CounterFactualProto` and `CEM` bug when the class is initialized a second time it wouldn't run as the TF graph would become disconnected
- Provide more useful error messages if external data APIs are down
- Skip tests using external data APIs if they are down

## [v0.2.3](https://github.com/SeldonIO/alibi/tree/v0.2.3) (2019-07-29)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.2.2...v0.2.3)
### Added
- `gen_category_map` utility function to facilitate using AnchorTabular explainer
- Extend `CounterFactualProto` with a more flexible choice for prototypes using k closest encoded instances
- Allow user to specify a hard target class for `CounterFactualProto`
- Distributed tests usign `pytest-xdist` to overcome TF global session interfering with tests running in the same process
### Changed
- Sample datasets now return a `Bunch` object by default, bundling all necessary and optional attributes for each dataset
- Loading sample datasets are now invoked via the `fetch_` functions to indicate that a network download is being made
### Fixed
- Remove `Home` from docs sidebar as this was causing the sidebar logo to not show up on landing page

## [v0.2.2](https://github.com/SeldonIO/alibi/tree/v0.2.2) (2019-07-05)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.2.1...v0.2.2)
### Added
- `codecov` support to CI
### Fixed
- Remove lexemes without word vectors in `spacy` models for `AnchorTabular`. This suppresses `spacy` warnings and also make the method (and tests) run a lot faster.

## [v0.2.1](https://github.com/SeldonIO/alibi/tree/v0.2.1) (2019-07-02)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.2.0...v0.2.1)
### Changed
- Remove `Keras` and `seaborn` from install requirements and create optional `[examples]` `extras_require`
- Remove `python-opencv` dependency in favour of `PIL`
- Improve type checking with unimported modules - now requires `python>3.5.1`
- Add some tests for `alibi.datasets`

## [v0.2.0](https://github.com/SeldonIO/alibi/tree/v0.2.0) (2019-05-24)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.1.0...v0.2.0)

**New features:**

- Counterfactual instances [\#78](https://github.com/SeldonIO/alibi/pull/78) ([jklaise](https://github.com/jklaise))
- Prototypical counterfactuals [\#86](https://github.com/SeldonIO/alibi/pull/86) ([arnaudvl](https://github.com/arnaudvl))

**Implemented enhancements:**

- Return nearest not predicted class for trust scores [\#63](https://github.com/SeldonIO/alibi/issues/63)
- Migrate Keras dependency to tf.keras [\#51](https://github.com/SeldonIO/alibi/issues/51)
- Add warning when no anchor is found [\#30](https://github.com/SeldonIO/alibi/issues/30)
- add anchor warning [\#74](https://github.com/SeldonIO/alibi/pull/74) ([arnaudvl](https://github.com/arnaudvl))
- Return closest not predicted class for trust scores [\#67](https://github.com/SeldonIO/alibi/pull/67) ([arnaudvl](https://github.com/arnaudvl))

**Closed issues:**

- Build docs on Travis [\#70](https://github.com/SeldonIO/alibi/issues/70)
- High level documentation for 0.1 [\#37](https://github.com/SeldonIO/alibi/issues/37)
- Counterfactual explainers [\#12](https://github.com/SeldonIO/alibi/issues/12)

**Merged pull requests:**

- Update example [\#100](https://github.com/SeldonIO/alibi/pull/100) ([jklaise](https://github.com/jklaise))
- Revert "Don't mock keras for docs" [\#99](https://github.com/SeldonIO/alibi/pull/99) ([jklaise](https://github.com/jklaise))
- Don't mock keras for docs [\#98](https://github.com/SeldonIO/alibi/pull/98) ([jklaise](https://github.com/jklaise))
- Cf [\#97](https://github.com/SeldonIO/alibi/pull/97) ([jklaise](https://github.com/jklaise))
- Cf [\#96](https://github.com/SeldonIO/alibi/pull/96) ([jklaise](https://github.com/jklaise))
- Cf [\#95](https://github.com/SeldonIO/alibi/pull/95) ([jklaise](https://github.com/jklaise))
- Cf [\#94](https://github.com/SeldonIO/alibi/pull/94) ([jklaise](https://github.com/jklaise))
- Cf [\#92](https://github.com/SeldonIO/alibi/pull/92) ([jklaise](https://github.com/jklaise))
- Cf [\#90](https://github.com/SeldonIO/alibi/pull/90) ([jklaise](https://github.com/jklaise))
- Cf [\#88](https://github.com/SeldonIO/alibi/pull/88) ([jklaise](https://github.com/jklaise))
- Add return type for counterfactuals [\#87](https://github.com/SeldonIO/alibi/pull/87) ([jklaise](https://github.com/jklaise))
- prototypical counterfactuals [\#86](https://github.com/SeldonIO/alibi/pull/86) ([arnaudvl](https://github.com/arnaudvl))
- Remove unnecessary method, rename loss minimization [\#85](https://github.com/SeldonIO/alibi/pull/85) ([jklaise](https://github.com/jklaise))
- Cf [\#84](https://github.com/SeldonIO/alibi/pull/84) ([jklaise](https://github.com/jklaise))
- Fix linting and remove old statsmodels tests [\#82](https://github.com/SeldonIO/alibi/pull/82) ([jklaise](https://github.com/jklaise))
- Some style and test fixes [\#81](https://github.com/SeldonIO/alibi/pull/81) ([jklaise](https://github.com/jklaise))
- Influence functions current work [\#79](https://github.com/SeldonIO/alibi/pull/79) ([jklaise](https://github.com/jklaise))
- WIP: Counterfactual instances [\#78](https://github.com/SeldonIO/alibi/pull/78) ([jklaise](https://github.com/jklaise))
- Counterfactual work so far [\#77](https://github.com/SeldonIO/alibi/pull/77) ([jklaise](https://github.com/jklaise))
- Add additional Python versions to CI [\#73](https://github.com/SeldonIO/alibi/pull/73) ([jklaise](https://github.com/jklaise))
- Add building docs and the Python package in CI [\#72](https://github.com/SeldonIO/alibi/pull/72) ([jklaise](https://github.com/jklaise))
- Bump master version to 0.1.1dev [\#68](https://github.com/SeldonIO/alibi/pull/68) ([jklaise](https://github.com/jklaise))

## [v0.1.0](https://github.com/SeldonIO/alibi/tree/v0.1.0) (2019-05-03)
**Closed issues:**

- Migrate CI to Travis post release [\#46](https://github.com/SeldonIO/alibi/issues/46)
- Trust scores [\#39](https://github.com/SeldonIO/alibi/issues/39)
- Make explicit Python\>=3.5 requirement before release [\#18](https://github.com/SeldonIO/alibi/issues/18)
- Remove dependency on LIME [\#17](https://github.com/SeldonIO/alibi/issues/17)
- Set up CI [\#5](https://github.com/SeldonIO/alibi/issues/5)
- Set up docs [\#4](https://github.com/SeldonIO/alibi/issues/4)

**Merged pull requests:**

- Update theme\_overrides.css [\#66](https://github.com/SeldonIO/alibi/pull/66) ([ahousley](https://github.com/ahousley))
- Add logo and trustscore example [\#65](https://github.com/SeldonIO/alibi/pull/65) ([jklaise](https://github.com/jklaise))
- Readme [\#64](https://github.com/SeldonIO/alibi/pull/64) ([jklaise](https://github.com/jklaise))
- Trustscore MNIST example [\#62](https://github.com/SeldonIO/alibi/pull/62) ([arnaudvl](https://github.com/arnaudvl))
- Fix broken links to methods notebooks [\#61](https://github.com/SeldonIO/alibi/pull/61) ([jklaise](https://github.com/jklaise))
- Initial Travis integration [\#60](https://github.com/SeldonIO/alibi/pull/60) ([jklaise](https://github.com/jklaise))
- Add tensorflow to doc generation for type information [\#59](https://github.com/SeldonIO/alibi/pull/59) ([jklaise](https://github.com/jklaise))
- Add numpy as a dependency to doc building for type information [\#58](https://github.com/SeldonIO/alibi/pull/58) ([jklaise](https://github.com/jklaise))
- Autodoc mocking imports [\#57](https://github.com/SeldonIO/alibi/pull/57) ([jklaise](https://github.com/jklaise))
- Avoid importing library for version [\#56](https://github.com/SeldonIO/alibi/pull/56) ([jklaise](https://github.com/jklaise))
- Add full requirement file for documentation builds [\#55](https://github.com/SeldonIO/alibi/pull/55) ([jklaise](https://github.com/jklaise))
- Focus linting and type checking on the actual library [\#54](https://github.com/SeldonIO/alibi/pull/54) ([jklaise](https://github.com/jklaise))
- Trust score high level docs and exposing confidence in alibi [\#53](https://github.com/SeldonIO/alibi/pull/53) ([jklaise](https://github.com/jklaise))
- fix bug getting imagenet data [\#52](https://github.com/SeldonIO/alibi/pull/52) ([arnaudvl](https://github.com/arnaudvl))
- WIP: Flatten explainer hierarchy [\#50](https://github.com/SeldonIO/alibi/pull/50) ([jklaise](https://github.com/jklaise))
- Add missing version file [\#48](https://github.com/SeldonIO/alibi/pull/48) ([jklaise](https://github.com/jklaise))
- Fix package version from failing install [\#47](https://github.com/SeldonIO/alibi/pull/47) ([jklaise](https://github.com/jklaise))
- trust scores [\#44](https://github.com/SeldonIO/alibi/pull/44) ([arnaudvl](https://github.com/arnaudvl))
- WIP: High level docs [\#43](https://github.com/SeldonIO/alibi/pull/43) ([jklaise](https://github.com/jklaise))
- WIP: CEM and Anchor docs [\#40](https://github.com/SeldonIO/alibi/pull/40) ([arnaudvl](https://github.com/arnaudvl))
- WIP: CEM [\#36](https://github.com/SeldonIO/alibi/pull/36) ([arnaudvl](https://github.com/arnaudvl))
- Counterfactuals [\#34](https://github.com/SeldonIO/alibi/pull/34) ([gipster](https://github.com/gipster))
- Refactoring counterfactuals to split work [\#32](https://github.com/SeldonIO/alibi/pull/32) ([jklaise](https://github.com/jklaise))
- Counterfactuals [\#31](https://github.com/SeldonIO/alibi/pull/31) ([gipster](https://github.com/gipster))
- Clean up [\#29](https://github.com/SeldonIO/alibi/pull/29) ([jklaise](https://github.com/jklaise))
- Make minimum requirements versions for testing and CI [\#27](https://github.com/SeldonIO/alibi/pull/27) ([jklaise](https://github.com/jklaise))
- Add Python \>= 3.5 requirement [\#26](https://github.com/SeldonIO/alibi/pull/26) ([jklaise](https://github.com/jklaise))
- Change CI test commands to use correct dependencies [\#21](https://github.com/SeldonIO/alibi/pull/21) ([jklaise](https://github.com/jklaise))
- add anchor image [\#20](https://github.com/SeldonIO/alibi/pull/20) ([arnaudvl](https://github.com/arnaudvl))
- Anchor text [\#15](https://github.com/SeldonIO/alibi/pull/15) ([arnaudvl](https://github.com/arnaudvl))
- Add support for rendering notebooks using nbsphinx and nbsphinx-link [\#14](https://github.com/SeldonIO/alibi/pull/14) ([jklaise](https://github.com/jklaise))
- WIP: Sphinx configuration [\#11](https://github.com/SeldonIO/alibi/pull/11) ([jklaise](https://github.com/jklaise))
- Ignore missing mypy imports globally for now [\#10](https://github.com/SeldonIO/alibi/pull/10) ([jklaise](https://github.com/jklaise))
- Add mypy to CI and create code style guidelines [\#9](https://github.com/SeldonIO/alibi/pull/9) ([jklaise](https://github.com/jklaise))
- Flake8 setup [\#8](https://github.com/SeldonIO/alibi/pull/8) ([jklaise](https://github.com/jklaise))
- Initial CI & docs setup [\#6](https://github.com/SeldonIO/alibi/pull/6) ([jklaise](https://github.com/jklaise))
- Anchor [\#3](https://github.com/SeldonIO/alibi/pull/3) ([arnaudvl](https://github.com/arnaudvl))
- Create initial package skeleton [\#2](https://github.com/SeldonIO/alibi/pull/2) ([jklaise](https://github.com/jklaise))
- Add licence [\#1](https://github.com/SeldonIO/alibi/pull/1) ([jklaise](https://github.com/jklaise))



\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*

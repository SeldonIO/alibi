# Change Log

## [v0.5.8](https://github.com/SeldonIO/alibi/tree/v0.5.8) (2021-04-29)
[Full Changelog](https://github.com/SeldonIO/alibi/compare/v0.5.7...v0.5.8)

### Added
- Experimental explainer serialization support using `dill`. See [docs](https://docs.seldon.io/projects/alibi/en/latest/overview/saving.html) for more details.

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
- `ALE` was causing an error when a constant feature was present. This is now handled explicitly and the user has control over how to handle these features. See https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.ale.html#alibi.explainers.ale.ALE for more details.
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

# Change Log

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

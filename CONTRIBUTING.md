# Contributing Code

We welcome PRs from the community. This document outlines the standard
practices and development tools we use.

When you contribute code, you affirm that the contribution is your original work and that you license the work to the project under the project's open source license. Whether or not you state this explicitly, by submitting any copyrighted material via pull request, email, or other means you agree to license the material under the project's open source license and warrant that you have the legal authority to do so.

## Getting started
The easiest way to get started is to clone `alibi` and install it locally together with all the development dependencies
in a separate virtual environment:
```
git clone git@github.com:SeldonIO/alibi.git
cd alibi
pip install -e .
pip install -r requirements/dev.txt -r requirements/docs.txt
```
This will install everything needed to run alibi and all the dev tools
(docs builder, testing, linting etc.)

## Git pre-commit hooks
To make it easier to format code locally before submitting a PR, we provide
integration with `pre-commit` to run `flake8`, `mypy` and `pyupgrade` (via `nbqa`) hooks before every commit.
After installing the development requirements and cloning the package, run `pre-commit install`
from the project root to install the hooks locally. Now before every `git commit ...`
these hooks will be run to verify that the linting and type checking is correct. If there are
errors, the commit will fail and you will see the changes that need to be made.

## Testing
We use `pytest` to run tests.
Because `alibi` uses some TensorFlow 1.x constructs, to run all tests you need to invoke `pytest` twice as follows:
```bash
pytest -m tf1 alibi
pytest -m "not tf1 alibi"
```
[see also here](https://github.com/SeldonIO/alibi/blob/4d4f49e07263b20a25f552a8485844dc12281074/.github/workflows/ci.yml#L46-L47).
It is not necessary to run the whole test suite locally for every PR as this can take a long time, it is enough to run `pytest`
only on the affected test files or test functions. The whole test suite is run in CI on every PR.

Test files live together with the library files under `tests` folders.

Some tests use pre-trained models to test method convergence. These models and the dataset loading
functions used to train them live in the https://github.com/SeldonIO/alibi-testing repo which is
one of the requirements for running the test suite.

## Linter
We use `flake8` for linting adhering to PEP8 with exceptions defined in `setup.cfg`. This is run as follows:
```bash
flake8 alibi
```

## Type checking
We use type hints to develop the libary and `mypy` to for static type checking. Some
options are defined in `setup.cfg`. This is run as follows:
```bash
mypy alibi
```

## Docstrings
We adhere to the `numpy` style docstrings (https://numpydoc.readthedocs.io/en/latest/format.html)
with the exception of ommiting argument types in docstrings in favour of type hints in function
and class signatures. If you're using a `PyCharm`, you can configure this under
`File -> Settings -> Tools -> Python Integrated Tools -> Docstrings`.

#### Conventions

- Names of variables, functions, classes and modules should be written between single back-ticks.
     - ``` A `numpy` scalar type that ```
     - ``` `X` ```
     - ``` `extrapolate_constant_perc` ```

- Simple mathematical equations should be written between single back-ticks to facilitate readability in the console.
     - ``` A callable that takes an `N x F` tensor, for ```
     - ``` `x >= v, fun(x) >= target` ```

- Complex math should be written in LaTeX.
    - ``` function where :math:`link(output - expected\_value) = sum(\phi)` ```

- Variable values or examples of setting an argument to a specific values should be written in double back-ticks
to facilitate readability as they are rendered in a block with orange font-color.
   - ``` is set to ``True`` ```
   - ``` A list of features for which to plot the ALE curves or ``'all'`` for all features. ```
   - ``` The search is greedy if ``beam_size=1`` ```
   - ``` if the result uses ``segment_labels=(1, 2, 3)`` and ``partial_index=1``, this will return ``[1, 2]``. ```
   
- Listing the possible values an argument can take.
   - ``` Possible values are: ``'all'`` | ``'row'`` | ``None``. ```

- Returning the name of the variable and its description - standard convention and renders well. Writing the 
variable types should be avoided as it would be duplicated from variables typing.
```
Returns
-------
raw
    Array of perturbed text instances.
data
    Matrix with 1s and 0s indicating whether a word in the text has not been perturbed for each sample.
```

- Returning only the description. When the name of the variable is not returned, sphinx wrongly interprets the 
description as the variable name which will render the text in italic. If the text exceeds one line, ``` \ ``` need 
to be included after each line to avoid introducing bullet points at the beginning of each row. Moreover, if for 
example the name of a variable is included between single back-ticks, the italic font is canceled for all the words
with the exception of the ones inbetween single back-ticks.
```
Returns
-------
If the user has specified grouping, then the input object is subsampled and an object of the same \
type is returned. Otherwise, a `shap_utils.Data` object containing the result of a k-means algorithm \
is wrapped in a `shap_utils.DenseData` object and returned. The samples are weighted according to the \
frequency of the occurrence of the clusters in the original data.
```
 
- Returning an object which contains multiple attributes and each attribute is described individually. 
In this case the attribute name is written between single back-ticks and the type, if provided, would be written in 
double back-ticks.
```
Returns
-------
`Explanation` object containing the anchor explaining the instance with additional metadata as attributes. \
Contains the following data-related attributes

 - `anchor` : ``List[str]`` - a list of words in the proposed anchor.

 - `precision` : ``float`` - the fraction of times the sampled instances where the anchor holds yields \
 the same prediction as the original instance. The precision will always be  threshold for a valid anchor.

 - `coverage` : ``float`` - the fraction of sampled instances the anchor applies to.
```

- Documenting a dictionary follows the same principle the as above but the key should be written between 
double back-ticks.
```
Default perturbation options for ``'similarity'`` sampling

    - ``'sample_proba'`` : ``float`` - probability of a word to be masked.

    - ``'top_n'`` : ``int`` - number of similar words to sample for perturbations.

    - ``'temperature'`` : ``float`` - sample weight hyper-parameter if `use_proba=True`.

    - ``'use_proba'`` : ``bool`` - whether to sample according to the words similarity.
```

- Attributes are commented inline to avoid duplication.
```
class ReplayBuffer:
    """
    Circular experience replay buffer for `CounterfactualRL` (DDPG) ... in performance.
    """
    X: np.ndarray  #: Inputs buffer.
    Y_m: np.ndarray  #: Model's prediction buffer.
    ...
```

For more standard conventions, please check the [numpydocs style guide](https://numpydoc.readthedocs.io/en/latest/format.html).

## Building documentation
We use `sphinx` for building documentation. You can call `make build_docs` from the project root,
the docs will be built under `doc/_build/html`. Detail information about documentation can be found [here](doc/README.md).

## CI
All PRs triger a CI job to run linting, type checking, tests, and build docs. The CI script is located [here](https://github.com/SeldonIO/alibi/blob/master/.github/workflows/ci.yml) and should be considered the source of truth for running the various development commands.

## PR checklist
Checklist to run through before a PR is considered complete:
 - All functions/methods/classes/modules have docstrings and all parameters are documented.
 - All functions/methods have type hints for arguments and return types.
 - Any new public functionality is exposed in the right place (e.g. `explainers.__init__` for new explanation methods).  
 - [linting](#linter) and [type-checking](#type-checking) passes.
 - New functionality has appropriate [tests](#testing) (functions/methods have unit tests, end-to-end functionality is also tested).
 - The runtime of the whole test suite on [CI](#ci) is comparable to that of before the PR.
 - [Documentation](#building-documentation) is built locally and checked for errors/warning in the build log and any issues in the final docs, including API docs.
 - For any new functionality or new examples, appropriate links are added (`README.md`, `doc/source/index.rst`, `doc/source/overview/getting_started.md`,`doc/source/overview/algorithms.md`, `doc/source/examples`), see [Documentation for alibi](doc/README.md) for more information.
 - For any changes to existing algorithms, run the example notebooks manually and check that everything still works as expected and there are no extensive warnings/outputs from dependencies.
 - Any changes to dependencies are reflected in the appropriate place (`setup.py` for runtime dependencies, `requirements/dev.txt` for development dependencies, and `requirements/doc.txt` for documentation dependencies).


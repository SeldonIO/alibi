import copy

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_PD, DEFAULT_DATA_PD


class PartialDependence(Explainer):
    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None):
        """
        Partial Dependence for tabular datasets. Supports one feature or two feature interactions.

        Parameters
        ----------
        predictor
            A callable that takes in an `N x F` array as input and outputs an `N x T` array (`N` -number of data
            points, `F` - number of features, `T` - number of outputs/targets.
        feature_names
            A list of feature names used for displaying results.
        target_names
            A list of target/output names used for displaying results.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PD))

        self.predictor = predictor
        self.feature_names = feature_names
        self.target_names = target_names

    def explain(self,
                X: np.ndarray,
                features: List[Union[int, Tuple[int, int]]],
                target: int,
                response_method: Literal['auto', 'predict_proba', 'decision_function'] = 'auto',
                percentiles: Tuple[float, float] = (0.05, 0.95),
                grid_resolution: int = 100,
                method: Literal['auto', 'recursion', 'brute'] = 'auto',
                kind: Literal['average', 'individual', 'both'] = 'average') -> Explanation:
        """
        Calculate the Partial Dependence for each feature with respect to the given target and the dataset `X`.

        Parameters
        ----------
        X
            An `N x F` tabular dataset used to calculate Partial Dependence curves. This is typically the training
            dataset or a representative sample.
        features
            Features for which to calculate the Partial Dependence.
        target
            Single output target for multi-output problems.
        response_method
            Specifies the prediction function to be used. For classifier it specifies whether to use the
            `predict_proba` or the `decision_function`. For a regressor, the parameter is ignored. If set to `auto`,
            the `predict_proba` is tried first, and if not supported then it reverts to `decision_function`. Note
            that if `method='recursion'`, the that the prediction function always uses `decision_function`.
        percentiles
            Lower and upper percentiles used to create extreme values which can potential remove outliers in low
            density regions. The values must be in [0, 1].
        grid_resolution
            Number of equidistant points to split the range of each target feature.
        method
            The method used to calculate the average predictions

             - `'recursion'` - a faster alternative only supported by some tree-based model. For a classifier, the
             target response is always the decision function and NOT the predicted probabilities. Furthermore, since
             the `'recursion'` method computes implicitly the average of the Individual Conditional Expectation (ICE)
             by design, it is incompatible with ICE and the `kind` parameter must be set to `'average'`. Check the
             `sklearn documentation`_ for a list of supported tree-based classifiers.

            .. _sklearn documentation:
                https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html#sklearn.inspection.partial_dependence

             - `'brute'` - supported for any black-box prediction model, but is more computationally intensive.

             - `'auto'` - uses `'recursion'` if the `predictor` supports it. Otherwise uses the `'brute'` method.
        kind
            If set to `'average'`, then only the Partial Dependence (PD) averaged across all samples from the dataset
            is returned. If set to `individual`, then only the Individual Conditional Expectation (ICE) is returned for
            each individual from the dataset. Otherwise, if set to `'both'`, then both the PD and the ICE are returned.
            Note that for the faster `method='recursion'` option the only compatible paramter value is
            `kind='average'`. To plot the ICE, consider using the more computation intensive `method='brute'`.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the calculated Partial Dependece
            curves. See usage at `Partial Dependence examples`_ for details

            .. _Partial Dependence examples:
                https://docs.seldon.io/projects/alibi/en/latest/methods/PartialDependence.html
        """

    def _build_explanation(self):
        pass

    def reset_predictor(self, predictor: Any) -> None:
        pass
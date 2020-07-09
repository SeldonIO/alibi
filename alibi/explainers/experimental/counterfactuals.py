import copy
import logging

import numpy as np
import tensorflow as tf

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_CF
from alibi.explainers.base import get_implementation
from alibi.explainers.exceptions import CounterfactualError
from alibi.utils.frameworks import _check_tf_or_pytorch, infer_device
from alibi.utils.wrappers import get_blackbox_wrapper
from alibi.utils.gradients import numerical_gradients
from inspect import signature
from typing import Dict, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


WACTHER_CF_VALID_SCALING = ['median']
WACHTER_CF_PARAMS = ['scale_loss', 'constrain_features', 'feature_whitelist']


def _validate_loss_spec(method: str, predictor_type: str, loss_spec: Optional[dict]):

    # TODO: DOCSTRING
    if not loss_spec:
        return
    if method == 'wachter':
        _validate_wachter_loss_spec(loss_spec, predictor_type)
    return


def _validate_wachter_loss_spec(loss_spec, predictor_type):

    # TODO: ALEX: DOCSTRING
    # TODO: ALEX: TBD: SHOULD WE ENFORCE KWARGS IN SPEC?
    available_num_grad_fcns = set([list(numerical_gradients[key]) for key in numerical_gradients.keys()])

    if 'distance' not in loss_spec:
        raise CounterfactualError(f"Expected loss_spec to have key 'distance'. Found keys {loss_spec.keys}!")
    for term in loss_spec:
        if 'kwargs' not in term.keys():
            raise CounterfactualError(
                "Could not find keyword arguments for one of the loss terms. Each term in loss_spec is expected to "
                "have 'kwargs' entry, a dictionary with key-value pairs for keyword arguments of your function. "
                "If your function does not need keyword arguments, then set 'kwargs' : {}."
            )
    if predictor_type == 'blackbox':
        correct, grad_method = False, False
        grad_method_name = ''
        for term in loss_spec:
            if 'gradient_method' in term:
                grad_method = True
                grad_method_name = term['gradient_fcn']
            if 'grad_fn' in term and 'grad_fn_kwargs' in term:
                correct = True
                break
        if not correct:
            raise CounterfactualError(
                "When one of the loss terms that depends on the predictor cannot be differentiated, the loss "
                "specification must contain a term that has a key 'grad_fn' and a key 'grad_fn_kwargs'. grad_fn should"
                "be a callable that takes the same arguments as the corresponding 'fcn', and represents the derivative"
                "of the loss term with respect to the predictor. If the function takes no kwargs, then set "
                "'grad_fn_kwargs': {}"
            )
        if not grad_method:
            raise CounterfactualError(
                "For non-differentiable predictors, a 'gradient_method' must be specified in the loss specification. "
                "Available numerical differentiation methods are 'num_grad_batch'."
            )
        else:
            if grad_method_name not in available_num_grad_fcns:
                raise CounterfactualError(
                    f"Undefined numerical gradients calculation method. Avaialble methods are "
                    f"{available_num_grad_fcns}"
                )


def median_abs_deviation(X: np.ndarray) -> np.ndarray:
    """
    Computes the median of the feature-wise median absolute deviation from `X`
    """

    # TODO: ALEX: TBD: THROW WARNINGS IF THE FEATURE SCALE IS EITHER VERY LARGE OR VERY SMALL?
    # TODO: ALEX: TBD: MOVE TO UTILITY FILE, CREATE A SCALING DECORATOR, AUTOIMPORT THEN APPLY

    feat_median = np.median(X, axis=0)

    return np.median(np.abs(X - feat_median), axis=0)


def _select_features(X: np.ndarray, feature_whitelist: Union[str, np.ndarray]) -> np.ndarray:
    """
    Creates a mask that is used to select the input features to be optimised.

    Parameters
    ----------
    See `explain` documentation.
    """

    if isinstance(feature_whitelist, str):
        return np.ones(X.shape, dtype=X.dtype)

    if isinstance(feature_whitelist, np.ndarray):
        expected = X.shape
        actual = feature_whitelist.shape
        if X.shape != feature_whitelist.shape:
            raise ValueError(
                f"Expected feature_whitelist and X shapes to be identical but got {actual} whitelist dimension "
                f"and {expected} X shape!"
            )
        return feature_whitelist


class Counterfactual(Explainer):

    def __init__(self,
                 predictor,
                 predictor_type: str = 'blackbox',
                 method: str = 'wachter',
                 method_opts: Optional[dict] = None,
                 loss_spec: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework='tensorflow',
                 **kwargs
                 ):

        # TODO: ADD DOCSTRING
        super().__init__(meta=copy.deepcopy(DEFAULT_META_CF))
        if not _check_tf_or_pytorch(framework):
            raise ValueError(
                "Unknown framework specified for framework not installed. Please check spelling and/or install the "
                "framework in order to run this explainer."
            )
        # support for PyTorch models on different devices
        self.fitted = False
        self.framework = framework
        self.predictor = predictor
        model_device = kwargs.get('device', None)
        if not model_device:
            self.model_device = infer_device(predictor, predictor_type, framework)
        else:
            self.model_device = model_device

        self.loss_spec = loss_spec
        _validate_loss_spec(method, predictor_type, loss_spec)
        search_algorithm = get_implementation('counterfactual', method)
        # the black_box_wrapper converts inputs to np.ndarray before calling the predictor
        # and outputs to tensors after predict calls
        black_box_warpper = get_blackbox_wrapper(framework) if predictor_type == 'blackbox' else None
        self._explainer = search_algorithm(
            predictor,
            predictor_type=predictor_type,
            loss_spec=loss_spec,
            method_opts=method_opts,
            feature_range=feature_range,
            framework=framework,
            logger=logger,
            model_device=self.model_device,
            blackbox_wrapper=black_box_warpper,
        )
        self._explainer.optimizer.device = self.model_device

    def fit(self,
            X: Optional[np.ndarray] = None,
            scale: Union[bool, str] = False,
            constrain_features: bool = True) -> "Counterfactual":
        """
        Calling this method with an array of :math:`N` data points, assumed to be the leading dimension of `X`, has the
        following functions:

            - If `constrain_features=True`, the minimum and maximum of the array along the leading dimension constrain \
            the minimum and the maximum of the counterfactual
            - If the `scale` argument is set to `True` or `MAD`, then the distance between the input and the \
            counterfactual is scaled, feature-wise at each optimisiation step by the feature-wise median absolute \
            deviation (MAD) calculated from `X` as detailed in the notes. Other options might be supported in the \
            future (raise a feature request).

        Parameters
        ----------
        X
            An array of :math:`N` data points.
        scale
            If set to `True` or 'median', the MAD is computed and applied to the distance part of the function. 
        constrain_features
            If `True` the counterfactual features are constrained.

        Returns
        -------
        A fitted counterfactual explainer object.

        Notes
        -----
        If `scale='median'` then the following calculation is performed:

        .. math:: 

            MAD_{j} = \mathtt{median}_{i \in \{1, ..., n\}}(|x_{i,j} - \mathtt{median}_{l \in \{1,...,n\}}(x_{l,j})|)
        """  # noqa W605

        # TODO: A decorator-based soln similar to the numerical gradients can be implemented

        self._check_scale(scale)

        if X is not None:

            if self.scale:
                # infer median absolute deviation (MAD) and update loss
                if scale == 'median' or isinstance(scale, bool):
                    scaling_factor = median_abs_deviation(X)
                    self._explainer.optimizer.distance_fcn.keywords['feature_scale'] = scaling_factor

            if constrain_features:
                # infer feature ranges and update counterfactual constraints
                feat_min, feat_max = np.min(X, axis=0), np.max(X, axis=0)
                self._explainer.optimizer.cf_constraint = [feat_min, feat_max]

        self.fitted = True
        scaling_method = 'mad' if self.scale else 'N/A'
        params = {
            'scale_loss': self.scale,
            'scaling_method': scaling_method,
            'constrain_features': constrain_features,
            'fitted': self.fitted,
        }

        self._update_metadata(params, params=True)

        return self

    def _check_scale(self, scale: Union[bool, str]) -> None:
        """
        Checks whether scaling should be performed depending on user input.
        Parameters
        ----------
        scale
            User options for scaling.
        """

        scale_ = False  # type: bool
        if isinstance(scale, str):
            if scale not in WACTHER_CF_VALID_SCALING:
                logger.warning(f"Received unrecognised option {scale} for scale. No scaling will take place. "
                               f"Recognised scaling methods are: {WACTHER_CF_VALID_SCALING}.")
            else:
                scale_ = True

        if isinstance(scale, bool):
            if scale:
                logger.info(f"Defaulting to median absolute deviation scaling!")
                scale_ = True

        if scale_:
            loss_params = signature(self.loss_spec['distance']).parameters
            if 'feature_scale' not in loss_params:
                logger.warning(
                    f"Scaling option specified but the loss specified did not have a parameter named 'feature_scale'. "
                    f"Scaling will not be applied!"
                )
                scale_ = False

        self.scale = scale_  # type: bool

    def explain(self,
                X: np.ndarray,
                target_class: Union[str, int] = 'other',
                target_proba: float = 1.0,
                optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                optimizer_opts: Optional[Dict] = None,
                feature_whitelist: Union[str, np.ndarray] = 'all',
                logging_opts: Optional[Dict] = None,
                method_opts: Optional[Dict] = None) -> "Explanation":

        # TODO: DOCUMENTATION

        # override default settings with user input
        if method_opts:
            for key in method_opts:
                # TODO: ALEX: TBD: DO WE REALLY WANT THIS?
                if key == 'tol':
                    logger.warning(
                        "tol cannot be overridden at explain time. Please pass method_opts={'tol': 0.5} when "
                        "instantiating the explainer."
                    )
                else:
                    self._explainer._set_attributes(method_opts[key])
        if logging_opts:
            self._explainer.logging_opts.update(logging_opts)

        if self._explainer.logging_opts['verbose']:
            logging.basicConfig(level=logging.DEBUG)

        # select features to optimize
        optimized_features = _select_features(X, feature_whitelist)

        # search for a counterfactual by optimising loss starting from the original solution
        result = self._explainer.counterfactual(
            X,
            optimized_features,
            target_class,
            target_proba=target_proba,
            optimizer=optimizer,
            optimizer_opts=optimizer_opts,
        )

        return self._build_explanation(X, result)

    def _update_metadata(self, data_dict: dict, params: bool = False, allowed: Optional[set] = None) -> None:
        """
        This function updates the metadata of the explainer using the data from the `data_dict`. If the params option
        is specified, then each key-value pair is added to the metadata `'params'` dictionary only if the key is
        included in `CF_PARAMS`.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
        allowed
            Set containing the parameters allowed in the update.
        """

        if params:
            for key in data_dict.keys():
                if key not in allowed:
                    logger.warning(
                        f"Parameter {key} not recognised, ignoring."
                    )
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)

    def _build_explanation(self, X: np.ndarray, result: dict) -> Explanation:
        """
        Creates an explanation object and re-initialises the response to allow calling `explain` multiple times on
        the same explainer.
        """

        # create explanation object
        result['instance'] = X
        result['status'] = {'converged': True}
        if not result['cf']:
            result['status']['converged'] = False

        explanation = Explanation(meta=copy.deepcopy(self.meta), data=result)
        # reset response
        self._explainer.initialise_response()

        return explanation

# TODO: PyTorch support
#  - have self.device argument set properly and partial their tensor conversion methods
#  - decorators for black-box need to be parametrized

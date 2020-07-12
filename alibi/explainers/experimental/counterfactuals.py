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


def _validate_loss_spec(method: str, predictor_type: str, loss_spec: Optional[dict]) -> None:
    """
    A dispatcher function to a validation function that checks whether the loss specification has been set correctly
    given the algorithm and the predictor type:

    Parameters
    ----------
    method: {'watcher'}
        Describes the algorithm for which the loss spec is validated.
    predictor_type: {'blackbox', 'whitebox'}
        Indicates if the algorithm has access to the predictor parameters or can only obtain predictions.
    loss_spec
        A mapping containing the loss specification to be validated.

    Raises
    ------
    CounterfactualError
        If the loss specification is not correct. See method-specific functions for details.
    """

    if not loss_spec:
        return
    if method == 'wachter':
        _validate_wachter_loss_spec(loss_spec, predictor_type)
    return


def _validate_wachter_loss_spec(loss_spec: dict, predictor_type: str) -> None:
    """
    Validates loss specification for counterfactual search, with joint input and distance loss weight optimisation
    (referred to as `wachter`).

    Parameters
    ---------
    loss_spec, predictor_type:
        See `_validate_loss_spec`
    
    Raises
    ------
    CounterfactualError
            
        - If `loss_spec` does not have 'distance' key, where the function that computes the distance between the current \
        solution and the initial condition is stored
        - If any of the loss terms in the spec do not have a `kwargs` entry
        - For black-box predictors, if there is no term that defines 'grad_fn' and 'grad_fn_kwargs' which tell the 
        algorithm how to compute the gradient of the 'prediction' term in the loss wrt to the model output
        - For black-box predictors, if 'gradient_method', an entry which specifies a function that computes the gradient \
        of the model output wrt to input (numerically) is not specified
        - If the numerical gradient computation function specified is not in the `numerical_gradients` registry. 
    """  # noqa W605

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

    Parameters
    ----------
    X
        Input array.

    Returns
    -------
        An array containing the median of the feature-wise median absolute deviation.
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
    X, feature_whitelist
        See `alibi.explainers.experimental.Counterfactual.explain` documentation.
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
    """
    An interface to counterfactual explanations computed by multiple methods and implemented in different frameworks.
    """
    def __init__(self,
                 predictor,
                 predictor_type: str = 'blackbox',
                 method: str = 'wachter',
                 method_opts: Optional[dict] = None,
                 loss_spec: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework='tensorflow',
                 **kwargs) -> None:
        """
        predictor
            A model which can be implemented in TensorFlow, PyTorch, or a callable that can be queried in order to 
            return predictions. The object returned (which is a framework-specific tensor) should always be 
            two-dimensional. For example, a single-output classification model operating on a single input instance 
            should return a tensor or array with shape  `(1, 1)`. The explainer assumes the `predictor` returns 
            probabilities. In the future this explainer may be extended to work with regression models.
        predictor_type: {'blackbox', 'whitebox'}
            
            - 'blackbox' indicates that the algorithm does not have access to the model parameters (e.g., the predictor \
            is an API endpoint so can only be queried to return prediction). This argument should be used to search for \
            counterfactuals of non-differentiable models (e.g., trees, random forests) or models implemented in machine \
            learning frameworks other than PyTorch and TensorFlow.
            
            - 'whitebox' indicates that the model is implemented in PyTorch or TensorFlow and that the explainer has 
            access to the parameters, so that the automatic differentiation and optimization of the framework can be 
            leveraged in the search process.  
            
        method: {'wachter'}
            
            - 'wachter' indicates that the method based on `Wachter et al. (2017)`_ (pp. 854) will be used to compute \
            the result. :math:`lambda` is maximized via a bisection procedure. For a given `lambda`, the current
            solution is updated with the loss function gradient using a stochastic gradient descent algorithm, which
            can be parametrized as explained in the `explain` method documentation.

             .. _Wachter et al. (2017):
           https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf
        method_opts
            Used to update any of the method default settings. For the ``'wachter'`` method, these default to the values 
            along with their default values are specified in `alibi.explainers.base.counterfactuals`, 
            `WACHTER_METHOD_OPTS`. It is recommended that the user runs the algorithm with the default and then and uses 
            the TensorBoard display to adjust these parameters in cases of non-convergence or long-running explanations.
            These parameters by passing updated values to `explain`, as explained therein. 
        loss_spec
            This argument can be used to customize the terms of the loss function and a different specification is
            defined for each method. Moreover, the specification depends on `predictor_type`. See the documentation for
            `WACHTER_LOSS_SPEC_WHITEBOX` and `WACHTER_LOSS_SPEC_BLACKBOX` in the subpackage of `alibi.explainers.backend`
            corresponding to your framework for further details about how it can be customized and our `documentation`_
            for the loss that these specifications represent. Note that to change the loss function, all the terms have
            to be specified, partially complete specification are not supported.

            .. _documentation: https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html
        
        feature_range
            Tuple with upper and lower bounds for feature values of the counterfactual. The upper and lower bounds can
            be floats or numpy arrays with dimension :math:`(N, )` where `N` is the number of features, as might be the
            case for a tabular data application. For images, a tensor of the same shape as the input data can be applied
            for pixel-wise constraints. If `fit` is called with argument `X`, then  feature_range is automatically
            updated as `(X.min(axis=0), X.max(axis=0))`.
        framework: {'pytorch', 'tensorflow'}
            The framework in which the model is implemented for ``'whitebox'`` predictors, or the framework used to run
            the optimization for ``'blackbox'`` predictors. PyTorch and TensorFlow are optional dependencies so they
            must be installed before running this algorithm. PyTorch support will be available in future releases, only 
            ``'tensorfllow'`` is a valid version for the current release.
        kwargs
            Valid options:
                - device: placeholder for PyTorch GPU support. If not specified and a ``'whitebox'`` predictor is \
                specified for PyTorch, it is inferred from the device of the model parameters attribute.   
        """ # noqa W605

        # TODO (ONGOING): UPDATE LINKS IN DOCSTRING AS NEW METHODS ARE ADDED.

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
        blackbox_wrapper = get_blackbox_wrapper(framework) if predictor_type == 'blackbox' else None
        self._explainer = search_algorithm(
            predictor,
            predictor_type=predictor_type,
            loss_spec=loss_spec,
            method_opts=method_opts,
            feature_range=feature_range,
            framework=framework,
            logger=logger,
            model_device=self.model_device,
            blackbox_wrapper=blackbox_wrapper,
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
        """
        Find a  counterfactual :math:`X'` for instance :math:`X`, given the `target_class` and the desired probability
        predicted by the model for `target_class` when :math:`X'` is input. The probability is reached with tolerance
        `tol` (see `WACHTER_METHOD_OPTS` `here`_ for default value). The search procedure is guided by the value of the
        method argument passed to the explainer. For details regarding the optimization procedure for each method, refer
        to our `detailed`_ documentation of the implementations.

        .. _here  https://docs.seldon.io/projects/alibi/en/stable/api/alibi.explainers.base.counterfactuals.html
        .. _detailed https://docs.seldon.io/projects/alibi/en/stable/api/alibi.explainers.base.counterfactuals.html

        Parameters
        ----------
        X
             Instance for which a counterfactual is to be generated. Only 1 instance can be explained at one time, 
            so the shape of the `X` array is expected to be `(1, ...)` where the ellipsis corresponds to the dimension 
            of one datapoint. In the future, batches of instances may be supported via a distributed implementation.
        target_class: {'same', 'other', int}
            Target class for the counterfactual, :math:`t` in the equation above:

                - `'same'`: the predicted class of the counterfactual will be the same as the instance to be explained
                - `'other'`: the predicted class of the counterfactual will be the class with the closest probability to \
                the instance to be explained 
                - ``int``: an integer denoting desired class membership for the counterfactual instance.
        target_proba
            Target probability to predicted by the model given the counterfactual.
        optimizer
            This argument can be used in two ways:
            
                - To pass an initialized optimizer, which might be possible for TensorFlow users. If not specified, \
                the optimizer will default to ADAM optimizer with polynomial decay initialised as follows::
                
                    Adam(learning_rate=PolynomialDecay(0.1, max_iter, end_learning_rate=0.002, power=1))
                    
                Here `max_iter` is read from the default method options defined `here`_ and can be overriden via the 
                `method_opts` argument. 
                
                - To pass the an optimizer class. This class is initialialized using the keyword arguments specified in
                `optimizer_opts` by the backend implementation. This is necessary to allow PyTorch user to customize the
                optimizer. 
        optimizer_opts
            These options are used to initialize the optimizer if a class (as opposed to an optimizer instance) is 
            passed to the explainer, as might be the case if a PyTorch user wishes to change the default optimizer. 
        feature_whitelist
            Indicates the feature dimensions that can be optimised during counterfactual search. Defaults to `'all'`,
            meaning that all feature will be optimised. A numpy array of the same shape as `X` (i.e., with a leading 
            dimension of 1) containing `1` for the features to be optimised and `0` for the features that keep their 
            original values.
        logging_opts
            A dictionary that specifies any changes to the default logging options specified in 
            `CF_LOGGING_OPTS_DEFAULT`, with the following structure::
                
                {
                    'verbose': False,
                    'log_traces': True,
                    'trace_dir': None,
                    'summary_freq': 1,
                    'image_summary_freq': 10,
                    'tracked_variables': {'tags': [], 'data_types': [], 'descriptions': []},
                }
            
            The default value for the dictionary is documented in the `WACHTER_CF_LOGGING_OPTS_DEFAULT` documentation
            `here_`.
            
                - 'verbose': if `False` the logger will be set to ``INFO`` level 
                
                - 'log_traces': if `True`, data about the optimisation process will be logged with a frequency specified \
                by `summary_freq` input. Such data include the learning rate, loss function terms, total loss and the \
                information about :math:`\lambda`. The algorithm will also log images if `X` is a 4-dimensional tensor \
                (corresponding to a leading dimension of 1 and `(H, W, C)` dimensions), to show the path followed by the \
                optimiser from the initial condition to the final solution. The images are logged with a frequency \
                specified by `image_summary_freq`. For each `explain` run, a subdirectory of `trace_dir` with `run_{}` \
                is created and  {} is replaced by the run number. To see the logs run the command in the `trace_dir` \
                directory:: 
            
                    ``tensorboard --logdir trace_dir``
                 replacing ``trace_dir`` with your own path. Then run ``localhost:6006`` in your browser to see the \
                traces. The traces can be visualised as the optimisation proceeds and can provide useful information \
                on how to adjust the optimisation in cases of non-convergence.
                
                - 'trace_dir': the directory where the optimisation infromation is logged. If not specified when \
                `log_traces=True`, then the logs are saved under `logs/cf`. 
                
                - 'summary_freq': logging frequency for optimisation information.
                
                - 'image_summary_freq': logging frequency for intermediate counterfactuals (for image data).
        method_opts
            This contains the hyperparameters specific to the method used to search for the counterfactual. These are 
            documented in the base implementations for the specific algorithms and can be found `here`_.
        """  # noqa W605

        # override default method settings with user input
        if method_opts:
            for key in method_opts:
                if key == 'tol':
                    self._explainer._set_attributes({'tol': method_opts['tol']})
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


# TODO: Keep in mind:
#  - to update the docstrings if adding new methods. They often ref module names so should be updated accordingly.

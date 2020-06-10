import copy
import logging
import os

import numpy as np
import tensorflow as tf

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.utils.wrappers import methdispatch
from copy import deepcopy
from functools import partial
from inspect import signature
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


# TODO: ALEX: This is a simple draft to get the CF code to work. More generally, we would
#  define an object/series of functions to check if PyTorch/TF/both are installed and import
#  appropriate casting function. I think there might be a nice way to do this via decorators

def to_numpy_arr(X: Union[tf.Tensor, tf.Variable, np.ndarray]):
    """
    A function that casts an array-like object `X` to a `np.array` object.
    """
    if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
        return X.numpy()
    return X

# TODO: ALEX: TBD I think it's best practice to define all the custom
#  errors in one file (e.g., exceptions.py/error.py). We
#  could standardise this across the library


class CounterfactualError(Exception):
    pass


CF_SUPPORTED_DISTANCE_FUNCTIONS = ['l1']
CF_VALID_SCALING = ['MAD']
CF_PARAMS = ['scale_loss', 'constrain_features', 'feature_whitelist']
CF_LAM_OPTS_DEFAULT = {
    'lam_init': 0.1,
    'lams': None,
    'lam_cf_threshold': 5,
    'lam_multiplier': 10,
    'lam_divider': 10,
    'decay_steps': 10,
    'common_ratio': 10,
    'max_lam_steps': 10,
}
"""dict: The counterfactual search process depends on the hyperparameter :math:`\lambda`, as described in the `explain`
documentation below. The explainer will first try to determine a suitable range for :math:`\lambda` and find 
:math:`\lambda` using a bisection algorithm. This dictionary contains the default settings for these methods. 
See `explain` method documentation for an explanation of these parameters.

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys to
the explainer constructor or when calling `explain`. If the same subset of arguments is specified in both 
`explain` and the constructor, the `explain` options will override the constructor options.

Examples
--------
To override the `max_lam_steps` parameters at explain time `explain` should be called with::

    lam_opts = { 'max_lam_steps': 50}.
""" # noqa W605


CF_SEARCH_OPTS_DEFAULT = {
    'max_iter': 1000,
    'early_stop': 50,
}
"""dict: The default values governing the search process. See `explain` method for a detailed descriptions of each 
parameter.

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys to
the explainer constructor or when calling `explain`. If the same subset of arguments is specified in both 
`explain` and the constructor, the `explain` options will override the constructor options.

Examples
--------
To override the `early_stop` parameter at explain time, then `explain` should be called with::

    search_opts = {'early_stop': 100}.
"""
CF_ATTRIBUTES = set(CF_SEARCH_OPTS_DEFAULT.keys()) | set(CF_LAM_OPTS_DEFAULT.keys())


def counterfactual_loss(instance: tf.Tensor,
                        cf: tf.Variable,
                        lam: float,
                        feature_scale: tf.Tensor,
                        pred_probas: Optional[tf.Variable] = None,
                        target_probas: Optional[tf.Tensor] = None,
                        ) -> tf.Tensor:

    # TODO: ALEX: TBD: TO SUPPORT BLACK-BOX, PYTORCH WE NEED A SIMILAR FUNCTION AND A HIGHER LEVEL ROUTINE TO
    #  "DISPATCH" AMONG THESE FUNCTIONS

    # TODO: ALEX: DOCSTRING

    distance = lam * distance_loss(instance, cf, feature_scale)
    if pred_probas is None or target_probas is None:
        return distance
    pred = pred_loss(pred_probas, target_probas)
    return distance + pred


def distance_loss(instance: tf.Tensor,
                  cf: tf.Variable,
                  feature_scale,
                  distance_fcn: str = 'l1') -> tf.Tensor:

    # TODO: ALEX: DOCSTRING

    if distance_fcn not in CF_SUPPORTED_DISTANCE_FUNCTIONS:
        raise NotImplementedError(f"Distance function {distance_fcn} is not supported!")

    ax_sum = tuple(np.arange(1, len(instance.shape)))

    return tf.reduce_sum(tf.abs(cf - instance) / feature_scale, axis=ax_sum, name='l1')


def pred_loss(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING

    return tf.square(pred_probas - target_probas)


def range_constraint(
        X: Union[tf.Variable, tf.Tensor],
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray]) -> Union[tf.Variable, tf.Tensor]:
    """
    Clips `X` so that the its values lie within the :math:`[low, high]` interval. This function is used to constrain the
    values of the counterfactuals with the specified range.

    Parameters
    ----------
    X
        Tensor or Variable to be constrained.
    low, high
        If the input is of type `np.ndarray`, then its shape has to be such that it supports element-wise comparison
        with `X` is possible.

    Returns
    -------
    A tensor whose values are constrained.

    Raises
    ------
    ValueError:
        If `low` > `high`.
    """

    feat_interval = high - low
    if np.any(feat_interval < 0.):
        raise ValueError(
            f"Invalid constrain. Lower bound needs to be smaller upper bound but the difference was "
            f"{feat_interval} for an element.",
        )

    return tf.clip_by_value(X, clip_value_min=low, clip_value_max=high)


class Counterfactual(Explainer):
    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_fn: Callable,
                 shape: Tuple[int, ...],
                 loss_fn_kwargs: Optional[dict] = None,
                 tol: float = 0.5,
                 feature_range: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (-1e10, 1e10),
                 **kwargs) -> None:
        """
        Counterfactual explanation method based on `Wachter et al. (2017)`_ (pp. 854).
        
        .. _Wachter et al. (2017): 
           https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf

        Parameters
        ----------
        predictor
            Keras or TensorFlow model, assumed to return a tf.Tensor object that contains class probabilities. The object 
            returned should always be two-dimensional. For example, a single-output classification model operating on a 
            single input instance should return a tensor or array with shape  `(1, 1)`. In the future this explainer may 
            be extended to work with regression models.
        tol
            tolerance for the counterfactual target probability. THe algorithm will aim to ensure 
            :math:`|f_t(X') - p_t| \leq \mathtt{tol}`. Here :math:`f_t(X')` is the :math:`t`th output of the `predictor`
            on a proposed counterfactual `X'` and `p_t` is a target for said output, specified as `target_proba` when
            calling explain.
        shape
            Shape of input data. It is assumed of the form `(1, ...)` where the ellipsis replaces the dimensions of a
            data point. 
        feature_range
            Tuple with upper and lower bounds for feature values of the counterfactual. The upper and lower bounds can
            be floats or numpy arrays with dimension :math:`(N, )` where `N` is the number of features, as might be the
            case for a tabular data application. For images, a tensor of the same shape as the input data can be applied
            for pixel-wise constraints. If `fit` is called with argument `X`, then  feature_range is automatically
            updated as `(X.min(axis=0), X.max(axis=0))`.
        kwargs
            Accepted keyword arguments are `search_opts` and `lam_opts`, which can contain any subset of the keys of
            `CF_SEARCH_OPTS_DEFAULT` and `CF_LAM_OPTS_DEFAULT`, respectively. These should be used if the user whishes
            to override the default explainer settings upon initialisation. This may be employed, for instance, if the
            user has knowledge of good optimisation settings for their problem. The default settings can also be 
            overridden at `explain` time so it is not necessary to pass the settings to the constructor.  
    
        """  # noqa W605

        # TODO: ALEX: ADAPT DOCSTRING
        # TODO: ALEX: TBD: IF WE ALLOW CUSTOM LOSS, THEN DISTANCE_FN OPTION IS REDUNDANT - YOU CAN REWRITE THE LOSS
        #  YOURSELF

        super().__init__(meta=copy.deepcopy(DEFAULT_META_CF))
        # get params for storage in meta
        params = locals()
        remove = ['self', 'predictor', '__class__']
        for key in remove:
            params.pop(key)
        self.meta['params'].update(params)
        self.fitted = False

        self.predictor = predictor
        self.batch_size = shape[0]

        # create attribute and set them with default values
        self._set_attributes(CF_SEARCH_OPTS_DEFAULT)
        self._set_attributes(CF_LAM_OPTS_DEFAULT)
        # override defaults with user settings
        self._set_attributes(kwargs.get('search_opts', None))
        self._set_attributes(kwargs.get('lam_opts', None))
        # set a default optimizer
        self._set_default_optimizer()
        # used to reset the optimiser
        self._optimizer_copy = deepcopy(self.optimizer)

        # feature scale updated in fit if a dataset is passed
        self.feature_scale = tf.identity(1.0)
        # TODO: ALEX: TBD: I THINK THIS IS NOT A GOOD DESIGN, FOR REASONS WHICH WILL BECOME APPARENT LATER
        self.loss = loss_fn
        if loss_fn_kwargs is not None:
            self.loss = partial(self.loss, **loss_fn_kwargs)
        self.tol = tol

        # init. at explain time
        self.target_proba = None  # type: tf.Tensor
        self.instance = None  # type: tf.Tensor
        self.cf = None  # type: tf.Variable

        # initialisation method and constraints for counterfactual
        self.cf_constraint = partial(range_constraint, low=feature_range[0], high=feature_range[1])

        # scheduler and optimizer initialised at explain time
        self.optimizer = None
        self.step = 0
        self.lam_step = 0

        # return templates
        self._initialise_response()
        self._num_explain_calls = 0

        # check shape of data for logging purposes
        self._log_image = False
        if len(shape) == 4:
            self._log_image = True

    def _set_attributes(self, attrs: Union[dict, None]) -> None:
        """
        Sets the attributes of the explainer using the (key, value) pairs in attributes. Ignores attributes that
        are not specified in CF_ATTRIBUTES.
        """

        # called with None if the attributes are not overriden
        if not attrs:
            return

        for key, value in attrs.items():
            if key not in CF_ATTRIBUTES:
                logger.warning(f"Attribute {key} unknown. Ignoring ...")
                continue
            setattr(self, key, value)

    def _initialise_response(self) -> None:
        """
        Initialises the templates that will form the body of the `explanation.data` field.
        """

        # NB: (class, proba) could be renamed to (output_idx, output) to make sense for regression settings
        self.this_result = dict.fromkeys(
            ['X', 'distance_loss', 'prediction_loss', 'total_loss', 'lambda', 'step', 'proba', 'class']
        )
        self.search_results = copy.deepcopy(DEFAULT_DATA_CF)
        self.search_results['all'] = {i: [] for i in range(self.max_lam_steps)}
        self.search_results['lambda_sweep'] = {}

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
            If set to `True` or 'MAD', the MAD is computed and applied to the distance part of the function. 
        constrain_features
            If `True` the counterfactual features are constrained.
        
        Returns
        -------
        A fitted counterfactual explainer object.
        
        Notes
        -----
        If `scale='MAD'` then the following calculation is performed:

        .. math:: 
            
            MAD_{j} = \mathtt{median}_{i \in \{1, ..., n\}}(|x_{i,j} - \mathtt{median}_{l \in \{1,...,n\}}(x_{l,j})|)
        """ # noqa W605

        # TODO: ALEX: TBD  Allow pd.DataFrame in fit?
        # TODO: ALEX: TBD: Could allow an optional scaler argument s.t. people can use skelearn (this might require
        #  effort so I would put it on hold for now)
        # TODO: epsilons ?

        self._check_scale(scale)

        if X is not None:

            if self.scale:
                # infer median absolute deviation (MAD) and update loss
                if scale == 'MAD' or isinstance(scale, bool):
                    self._mad_scaling(X)
            if constrain_features:
                # infer feature ranges and update counterfactual constraints
                feat_min, feat_max = np.min(X, axis=0), np.max(X, axis=0)
                self.cf_constraint.keywords['low'] = feat_min
                self.cf_constraint.keywords['high'] = feat_max

        self.fitted = True

        params = {
            'scale_loss': self.scale,
            'constrain_features': constrain_features,
            'fitted': self.fitted,
        }

        self._update_metadata(params, params=True)

        return self

    def _mad_scaling(self, X: np.ndarray) -> None:
        """
        Computes feature-wise mean absolute deviation for `X`. Fixes the latter as an invariant input to the loss
        function, which divides the distance between the current solution and the instance whose counterfactual is
        to be found by this value.
        """

        # TODO: ALEX: TBD: THROW WARNINGS IF THE FEATURE SCALE IS EITHER VERY LARGE OR VERY SMALL?

        feat_median = np.median(X, axis=0)
        mad = np.median(np.abs(X - feat_median), axis=0)
        self.feature_scale = tf.identity(mad)
        self.loss = partial(self.loss, feature_scale=self.feature_scale)
        self.feature_scale_numpy = mad

    def _check_scale(self, scale: Union[bool, str]) -> None:
        """
        Checks whether scaling should be performed depending on user input.

        Parameters
        ----------
        scale
            User options for scaling.
        """

        scale_ = False
        if isinstance(scale, str):
            if scale not in CF_VALID_SCALING:
                logger.warning(f"Received unrecognised option {scale} for scale. No scaling will take place. "
                               f"Recognised scaling methods are: {CF_VALID_SCALING}.")
            else:
                scale_ = True

        if isinstance(scale, bool):
            if scale:
                logger.info(f"Defaulting to median absolute deviation scaling!")
                scale_ = True

        if scale_:
            loss_params = signature(self.loss).parameters
            if 'feature_scale' not in loss_params:
                logger.warning(
                    f"Scaling option specified but the loss specified did not have a parameter named 'feature_scale'. "
                    f"Scaling will not be applied!"
                )
                scale_ = False

        self.scale = scale_

    def explain(self,
                X: np.ndarray,
                target_class: Union[str, int] = 'other',
                target_proba: float = 1.0,
                optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                search_opts: Optional[dict] = None,
                lam_opts: Optional[dict] = None,
                feature_whitelist: Union[str, np.ndarray] = 'all',
                verbose: bool = False,
                log_traces: bool = True,
                trace_dir: str = None,
                summary_freq: int = 1,
                image_summary_freq: int = 10,
                ) -> Explanation:
        """
        Returns :math:`X'`, a counterfactual of `X` and information about the search process. The counterfactual is 
        computed by optimising the function 
        
        .. math:: \ell(X', X, \lambda) = {(f_t(X') - p_t)}^2 + \lambda \ell_1 (X', X)
        
        where :math:`f` is a predictor, :math:`t` is an index into the model output vector that returns the target 
        output for the counterfactual, :math:`p_t` is the target probability for the counterfactual and :math:`\ell_1`
        is the L1 norm.  
        
        The counterfactual is first optimised using gradient descent on the loss function. :math:`\lambda` is optimised
        using a bisection algorithm, as follows:
        
            - if the number of counterfactuals exceeds `lam_cf_threshold`, then:

                * the :math:`\lambda` lower bound, :math:`\lambda_{lb}`, is set to :math:`\max(\lambda, \lambda_{lb})`
                * if :math:`\lambda < 10^9` then :math:`\lambda \gets 0.5*(\lambda_{lb}, \lambda_{ub})`
                * else :math:`\lambda` is mutiplied by `lambda_multiplier`

            - if the number of counterfactuals is below `lam_cf_threshold`, then:

                * the :math:`\lambda` upper bound, :math:`\lambda_{ub}`, is set to :math:`\min(\lambda, \lambda_{ub})`
                * if :math:`\lambda > 0` then :math:`\lambda \gets 0.5*(\lambda_{lb}, \lambda_{ub})`
                * else :math:`\lambda` is divided by `lambda_divider`

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
            Target probability predicted by the model given the counterfactual, :math:`p_t` in the equation above.
        optimizer:
            An initialised optimizer. If not specified, the optimizer will default to `tf.keras.optimizers.Adam`,
            initialised as follows::
                
                Adam(learning_rate=PolynomialDecay(0.1, max_iter, end_learning_rate=0.002, power=1))
                
            Here `max_iter` is read from `CF_SEARCH_DEFAULT` and can be changed using the `search_opts` kwarg, as 
            described in the `CF_SEARCH_DEFAULT` documentation. This argument can be used to adjust the optimizer if
            the default one fails to converge to an appropriate solution.
        search_opts
            A dictionary that controls the options for the optimisation process, with the following structure::
            
                {
                    'max_iter': 1000,
                    'early_stop': 50,
                }
                            
            The default value for the dictionary is documented in the `CF_SEARCH_OPTS_DEFAULT` documentation. The keys  
            represent:
            
                - 'max_iter': Maximum number of iterations to run the gradient descent for (number of inner loops for \
                each outer loop)
                - 'early_stop': the inner loop will terminate after this number of iterations if either no solutions \
                satisfying the constraint on the prediction are found or if a solution is found at every step for this \
                amount of steps.                
        lam_opts
            A dictionary that controls the optimisation process for :math:`\lambda`, with the following structure::
            
                {
                   'lam_init': 0.1,
                    'lams': None,
                    'lam_cf_threshold': 5,
                    'lam_multiplier': 10,
                    'lam_divider': 10,
                    'decay_steps': 10,
                    'common_ratio': 10,
                    'max_lam_steps': 10,
                }
            
            The default value for the dictionary is document in the `CF_LAM_OPTS_DEFAULT` documentation. The keys  
            represent:
                
                - 'lam_init': :math:`\lambda` in the equation above. Can be interpreted as a regularisation constant \
                on the prediction
                - 'lams': counterfactuals exist in restricted regions of the optimisation space, and finding these \
                regions depends on the :math:`\lambda` paramter. The algorithm first runs an optimisation loop to \
                determine if counterfactuals  exist for a given :math:`\lambda`. The default sequence of \
                :math:`\lambda` s is:: 
            
                        lams = np.array([lam_init / common_ratio ** i for i in range(decay_steps)]) 
                 This sequence can be overriden by passing lams directly. For each :math:`\lambda` step, \
               ``max_iter // decay_steps`` iterations of gradient descent updates are performed. `common_ratio` \
                and `decay_steps` default values are documented in the `CF_LAM_OPTS_DEFAULT` documentation
                
                - 'lam_cf_threshold': a threshold on the number of counterfactuals that satisfy the target constraint \
                above which :math:`\lambda` is increased
                - 'lam_multiplier', 'lam_divider': the factors by which multiplied/divided when the when the number of \
                solutions which satisfy the constraint exceeds/is below `lam_cf_threshold` and certain conditions (see \
                 above) on :math:`\lambda` are satisfied
                - 'decay_steps': overrides the default len for `lams` and the number of gradient updates during \
                :math:`\lambda` search, allowing the user to customise the exponential decay
                - 'common_ratio': overrides the eponential decay schedule for :math:`\lambda`, allowing the user to \
                customise the default exponential decay
                - 'max_lam_steps': maximum number of times to adjust the regularization constant before terminating \
                the search (number of outer loops).
        feature_whitelist
            Indicates the feature dimensions that can be optimised during counterfactual search. Defaults to `'all'`,
            meaning that all feature will be optimised. A numpy array of the same shape as `X` (i.e., with a leading 
            dimension of 1) containing `1` for the features to be optimised and `0` for the features that keep their 
            original values.
        verbose
            If `False` the logger will be set to ``INFO`` level. 
        log_traces
            If `True`, data about the optimisation process will be logged with a frequency specified by `summary_freq`
            input. Such data include the learning rate, loss function terms, total loss and the information about 
            :math:`\lambda`. The algorithm will also log images if `X` is a 4-dimensional tensor (corresponding to a 
            leading dimension of 1 and `(H, W, C)` dimensions), to show the path followed by the optimiser from the 
            initial condition to the final solution. The images are logged with a frequency specified by 
            `image_summary_freq`. For each `explain` run, a subdirectory of `trace_dir` with `run_{}` is created and 
            {} is replaced by the run number. To see the logs run the command in the `trace_dir` directory 
            
                ``tensorboard --logdir trace_dir``
                
            replacing ``trace_dir`` with your own path. Then run ``localhost:6006`` in your browser to see the traces.
            The traces can be visualised as the optimisation proceeds and can provide useful information on how to
            adjust the optimisation in cases of non-convergence.
        trace_dir
            The directory where the optimisation infromation is logged. If not specified when `log_traces=True`, then
            the logs are saved under `logs/cf`. 
        summary_freq
            Logging frequency for optimisation information.
        image_summary_freq
            Logging frequency for intermediate counterfactuals (for image data).
            
        Returns
        -------
        An `Explanation` object containing the counterfactual with additional metadata.

        Raises
        ------
        CounterfactualError
            If the leading dimension of the data point fed is not 1.
        ValueError
            If the `feature_whitelist` argument is a numpy array that has a different shape to `X`.
        """  # noqa W605

        # TODO: ALEX: TBD: THIS IS RELATIVELY GENERIC, SHOULDN'T HAVE TO CHANGE. EXCEPT THAT OTHER METHODS MAY
        #  SUPPORT BATCH SO THEN WE WOULD HAVE TO MOVE THE ERROR INSIDE _SEARCH.

        # TODO: In the future, the docstring should reference to documentation where the structure of the response
        #  is detailed

        # check inputs
        if X.shape[0] != 1:
            raise CounterfactualError(
               f"Only single instance explanations supported (leading dim = 1), but got leading dim = {X.shape[0]}",
            )

        # override default settings with user settings
        self._set_attributes(search_opts)
        self._set_attributes(lam_opts)

        # TODO: ALEX: CHECK THIS WORKS
        # set verbosity
        if not verbose:
            logger.setLevel(logging.WARNING)

        # select features to optimise
        self._create_gradient_mask(X, feature_whitelist)

        # setup Tensorboard
        self._setup_tb(log_traces, trace_dir, summary_freq=summary_freq, image_summary_freq=image_summary_freq)

        self.instance_numpy = X
        self.target_proba_numpy = np.array([target_proba])[:, np.newaxis]

        if optimizer is not None:
            self._reset_optimizer(optimizer)

        # make a prediction
        Y = self._make_prediction(X)
        instance_class = self._get_label(to_numpy_arr(Y))
        instance_proba = to_numpy_arr(Y[:, instance_class])

        # helper function to return the model output given the target class
        # TODO: ALEX: TBD: SPECIFIC ASSUMPTION ABOUT WHAT GOES IN THE LOSS FUNCTION
        self.get_cf_prediction = partial(self._get_cf_prediction, instance_class=instance_class)
        # initialize optimised variables, targets and loss weight
        self._initialise_variables(X, target_class, target_proba)

        # search for a counterfactual by optimising loss starting from the original solution
        result = self._search(init_cf=X)
        self._reset_step()

        return self._build_explanation(X, result, instance_class, instance_proba)

    def _create_gradient_mask(self, X: np.ndarray, feature_whitelist: Union[str, np.ndarray]) -> None:
        """
        Creates a gradient mask used to keep the values of specified features constant. The value is cast to a tensor
        in `_initiaize_variables` to avoid making this function framework-specific.

        Parameters
        ----------
        See `explain` documentation.
        """

        # TODO: ALEX: TBD: GENERIC

        if isinstance(feature_whitelist, str):
            self._mask = np.ones(X.shape)

        if isinstance(feature_whitelist, np.ndarray):
            expected = X.shape
            actual = feature_whitelist.shape
            if X.shape != feature_whitelist.shape:
                raise ValueError(
                    f"Expected feature_whitelist and X shapes to be identical but got {actual} whitelist dimension "
                    f"and {expected} X shape!"
                )
            self._mask = feature_whitelist

    def _setup_tb(self, log_traces: bool, trace_dir: str, summary_freq: int = 1, image_summary_freq: int = 10) -> None:
        """
        Creates a summary file writer for the current explain call. Sets `_logging` attribute.

        Parameters
        ----------
        log_traces
            Whether information about the optimisation should be stored.
        trace_dir
            The root directory where sub-directories containing event files are saved.
        summary_freq
            Scalar logging frequency.
        image_summary_freq
            Image logging frequency.
        """

        # TODO: ALEX: TBD: GENERIC

        self._num_explain_calls += 1
        self._logging = False
        if log_traces:
            self._logging = True
            if not trace_dir:
                trace_dir = 'logs/cf'
            trace_dir = os.path.join(trace_dir, f"run_{self._num_explain_calls}")
            self.writer = tf.summary.create_file_writer(trace_dir)
            self.summary_freq = summary_freq
            self.image_summary_freq = image_summary_freq

    def _make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Makes a prediction for data points in `X`.

        Parameters
        ----------
        X
            A tensor of arrays for which predictions are requested.
        """

        # TODO: ALEX: TBD: GENERIC

        return self.predictor(X, training=False)

    @staticmethod
    def _get_label(Y: np.ndarray, threshold: float = 0.5) -> int:
        """
        Given a prediction vector containing the model output for a prediction on a single data point,
        it returns an integer representing the predicted class of the data point (the label). Accounts for the case
        where a binary classifier returns a single output.

        Parameters
        ----------
        Y
            Array of shape `(1, N)` where N is the number of model outputs. Contains probabilities.
        threshold
            The threshold used to convert a probability to a class label.

        Returns
        -------
        An integer representing the predicted label.
        """

        # TODO: ALEX: TBD: THIS MAY HAVE TO BE GENERALISED IF SOME METHODS SUPPORT BATCH AND OUR X is not
        #  GUARANTEED TO BE OF SHAPE (1, ...)
        # TODO: ALEX: TBD: Parametrise `threshold` in explain or do we assume people always assign labels on this rule?

        if Y.shape[1] == 1:
            return int(Y > threshold)
        else:
            return np.argmax(Y)

    def _set_default_optimizer(self) -> None:
        """
        Initialises the explainer with a default optimizer.
        """

        # TODO: ALEX: TBD: GENERIC ALTHOUGH MAY WANT TO SPECIFY THE CONFIG ACTUALLY AND NOT HAVE
        #  THIS HARDCODED LIKE THIS.

        self.lr_schedule = PolynomialDecay(0.1, self.max_iter, end_learning_rate=0.002, power=1)
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    def _reset_optimizer(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """
        Resets the default optimizer. It is used to set the optimizer specified by the user as well as reset the 
        optimizer at each :math:`\lambda` optimisation cycle.
        
        Parametrs
        ---------
        optimizer
            This is either the opimizer passed by the user (first call) or a copy of it (during `_search`).
        """  # noqa W605

        # TODO: METHOD SPECIFIC, SINCE WE ONLY REQUIRE THIS DUE TO LAMBDA IMPLEMENTATION

        self._optimizer_copy = deepcopy(optimizer)
        self.optimizer = optimizer

    @methdispatch
    def _get_cf_prediction(self, target_class: Union[int, str], cf: tf.Tensor, instance_class: int) -> tf.Tensor:
        """
        Returns the slice of the model output tensor that corresponds to the target class for the counterfactual that
        was specified by the user. See registered functions for details.

        Parameters
        ----------
        target_class
            The class that the model should predict on the counterfactual.
        cf
            Counterfactual.
        instance_class
            The class predicted on the instance with counterfactual cf.

        Returns
        -------
        A (1, 1) shape tensor containing the model prediction for `target_class` given `cf`.
        """

        # TODO: ALEX: TBD: WHETHER THIS IS GENERIC OR NOT, IT DEPENDS ON HOW WE IMPLEMENT THE REGRESSION VERSION

        raise TypeError(f"Expected target_class to be str or int but target_class was {type(target_class)}!")

    @_get_cf_prediction.register
    def _(self, target_class: str, cf: tf.Tensor, instance_class: int) -> tf.Tensor:
        """
        Returns a slice of the output tensor when target is a string.

        Parameters
        ----------
        target_class: {'same', 'other'}
            The target class for the counterfactual.
        cf
            Current solution.
        instance_class
            The class of the instance whose counterfactual is seached.

        Returns
        -------
        A slice from the model output tensor. If target is 'same', it is obtained using the `instance_class` index.
        Otherwise, a slice corresponding to the output with the highest probability other than `instance_class` is
        returned.
        """
        # TODO: ALEX: CHECK: Is type of cf correct?

        prediction = self._make_prediction(cf)
        if target_class == 'same':
            return prediction[:, instance_class]

        _, indices = tf.math.top_k(prediction, k=2)
        if indices[0][0] == instance_class:
            cf_prediction = prediction[:, indices[0][1].numpy()]
        else:
            cf_prediction = prediction[:, indices[0][0].numpy()]
        return cf_prediction

    @_get_cf_prediction.register
    def _(self, target_class: int, cf: tf.Tensor, instance_class: int) -> tf.Tensor:
        """
        Returns the slice from the model output indicated by `target_class`.
        """

        return self.predictor(cf, training=False)[:, target_class]

    def _initialise_variables(self, X: np.ndarray, target_class: int, target_proba: float) -> None:
        """
        Initialises optimisation variables so that the TensorFlow auto-differentiation framework
        can be used for counterfactual search.

        Parameters
        ----------
        X
            See `explain` documentation.
        target_class
            See `explain` documentation.
        target_proba
            See `explain` documentation.
        """

        # TODO: ALEX: TBD: I GUESS THIS IS GENERIC FOR STUFF THAT USES TENSORFLOW.
        #  TODO: ALEX: TBD: NOT SURE HOW THAT WOULD PLAY FOR REGRESSION

        # tf.identity is the same as constant but does not always create tensors on CPU
        self.target_proba = tf.identity(target_proba * np.ones(self.batch_size, dtype=X.dtype), name='target_proba')
        self.target_class = target_class
        self.instance = tf.identity(X, name='instance')
        self._initialise_cf(X)
        self._mask = tf.identity(self._mask, name='gradient mask')

    def _initialise_cf(self, X: np.ndarray) -> None:
        """
        Initializes the counterfactual to the data point `X`.

        Parameters
        ----------
        X
            Instance whose counterfactual is to be found.
        """

        # TODO: ALEX: TBD: GENERIC FOR TF BASED METHODS

        self.cf = tf.Variable(
            initial_value=X,
            trainable=True,
            name='counterfactual',
            constraint=self.cf_constraint,
        )

    def _search(self, *, init_cf: np.ndarray) -> dict:
        """
        Searches a counterfactual given an initial condition for the counterfactual. The search has two loops:
        
            - An outer loop, where :math:`\lambda` (the weight of the distance between the current counterfactual
            and the input `X` to explain in the loss term) is optimised using bisection
        
            - An inner loop, where for constant `lambda` the current counterfactual is updated using the gradient 
            of the counterfactual loss function. 
            
        Parameters
        ----------
        init_cf
            Initial condition for the optimisation.
        
        Returns
        -------
        A dictionary containing the search results, as defined in `alibi.api.defaults`.
        """  # noqa: W605

        # TODO: ALEX: TBD: THIS WHOLE METHOD IS SPECIFIC TO AN ALGORITHM, I DO NOT SEE THE POINT OF HAVING A GENERINC
        #  LOSS FCN. IF WE WANTED A DIFFERENT FLAVOUR OF OPTIMISATION, WE'D HAVE TO OVERRIDE IT

        lam_dict = self._initialise_lam(lams=self.lams, decay_steps=self.decay_steps, common_ratio=self.common_ratio)
        init_lam, init_lb, init_ub = lam_dict['midpoint'], lam_dict['lb'], lam_dict['ub']
        summary_freq = self.summary_freq
        cf_found = np.zeros((self.max_lam_steps, ), dtype=np.uint16)
        # re-init. cf as initial lambda sweep changed the initial condition
        self._initialise_cf(init_cf)

        self.lam, self.lam_lb, self.lam_ub = init_lam, init_lb, init_ub
        for lam_step in range(self.max_lam_steps):
            # re-set learning rate
            self._reset_optimizer(self._optimizer_copy)
            found, not_found = 0, 0
            for gd_step in range(self.max_iter):
                self._cf_step()
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self._make_prediction(self.cf)

                # save and optionally display results of current gradient descent step
                # TODO: ALEX: TBD: We could make `lam` AND `step` object properties and not have to pass this
                #  current state to the functions, but maybe it is a bit clearer what happens?
                current_state = (self.step, self.lam, to_numpy_arr(self.cf), to_numpy_arr(cf_prediction))
                write_summary = self.step % summary_freq == 0 and self._logging
                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_search_result(*current_state)
                    found += 1
                    not_found = 0
                else:
                    found = 0
                    not_found += 1
                    if write_summary:
                        self._collect_step_data(*current_state)

                # TODO: ALEX: TBD: THIS IS METHOD SPECIFIC
                if write_summary:
                    self._write_tb(
                        self.step,
                        self.lam,
                        self.lam_lb,
                        self.lam_ub,
                        cf_found,
                        prefix='counterfactual_search/')
                self.step += 1

                # early stopping criterion - if no solutions or enough solutions found, change lambda
                if found >= self.early_stop or not_found >= self.early_stop:
                    break

            self._bisect_lambda(
                cf_found,
                lam_step,
                lam_cf_threshold=self.lam_cf_theshold,
                lam_multiplier=self.lam_multiplier,
                lam_divider=self.lam_divider,
            )
            self.lam_step += 1

        self._display_solution(prefix='best_solutions/')

        return deepcopy(self.search_results)

    def _display_solution(self, prefix: str = '') -> None:
        """
        Displays the instance along with the counterfactual with the smallest distance from the instance.

        Parameters
        ----------
        prefix
            A string `used to place the solution into a separate Tensorboard tab.
        """

        # TODO: ALEX: TBD: GENERIC

        if not self._logging:
            return

        with tf.summary.record_if(self._log_image):
            with self.writer.as_default():
                tf.summary.image(
                    '{}counterfactuals/optimal_cf'.format(prefix),
                    self.search_results['cf']['X'],
                    step=0,
                    description=r"A counterfactual `X'` that satisfies `|f(X') - y'| < \epsilon` and minimises "
                                r"`d(X, X')`. Here `y'` is the target probability and `f(X')` is the model prediction "
                                r"on the counterfactual. The weight of distance part of the loss is {:.5f}"
                        .format(self.search_results['cf']['lambda']) # noqa
                )

                tf.summary.image(
                    '{}counterfactuals/original_input'.format(prefix),
                    self.instance,
                    step=0,
                    description="Instance for which a counterfactual is to be found."
            )

        self.writer.flush()

    def _reset_step(self):
        """
        Resets the optimisation step for gradient descent and for the weight optimisation step (`lam_step`)
        """
        self.step = 0
        self.lam_step = 0

        # TODO: ALEX: TBD: GENERIC

    def _initialise_lam(self,
                        lams: Optional[np.ndarray] = None,
                        decay_steps: int = 10,
                        common_ratio: int = 10) -> Dict[str, float]:
        """
        Runs a search procedure over a specified range of lambdas in order to determine a good initial value for this
        parameter. If `lams` is not specified, a range of lambda is created by diving an initial value by the geometric
        sequence ``[decay_factor**i for i in range(decay_steps)]``. The method saves the results of this sweep in the
        `'lambda_sweep'` field of the explanation 'data' field.

        Parameters
        ----------
        lams
            Array of shape ``(N,)`` containing values to try for `lam`.
        decay_steps
            The number of steps in the geometric sequence.
        common_ratio
            The common ratio of the geometric sequence.

        Returns
        -------
        A dictionary with keys:

            - 'lb': lower bound for lambda
            - 'ub': upper bound for lambda
            - 'midpoint': the midpoints of the interval [lb, ub]
        """

        # TODO: ALEX: TBD: SPECIFIC TO A PARTICULAR SEARCH METHOD

        n_steps = self.max_iter // decay_steps
        if lams is None:
            lams = np.array([self.lam_init / common_ratio ** i for i in range(decay_steps)])  # exponential decay
        cf_found = np.zeros(lams.shape, dtype=np.uint16)

        logger.debug('Initial lambda sweep: %s', lams)

        for lam_step, lam in enumerate(lams):
            # optimiser is re-created so that lr schedule is reset for every lam
            self._reset_optimizer(self._optimizer_copy)
            for gd_step in range(n_steps):
                # update cf with loss gradient for a fixed lambda
                self._cf_step()
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self._make_prediction(self.cf)

                # save search results and log to TensorBoard
                write_summary = self._logging and self.step % self.summary_freq == 0
                current_state = (self.step, lam, to_numpy_arr(self.cf), to_numpy_arr(cf_prediction))
                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_search_result(*current_state)
                else:
                    if write_summary:
                        self._collect_step_data(*current_state)

                if write_summary:
                    self._write_tb(self.step, lam, 0, 0, cf_found, prefix='lambda_sweep/')
                self.step += 1
            self.lam_step += 1

        self._reset_step()

        logger.debug(f"Counterfactuals found: {cf_found}")

        # determine upper and lower bounds given the solutions found
        lb_idx_arr = np.where(cf_found > 0)[0]
        if len(lb_idx_arr) < 2:
            logger.exception("Could not find lower bound for lambda, try decreasing lam_init!")
            raise CounterfactualError("Could not find lower bound for lambda, try decreasing lam_init!")
        lam_lb = lams[lb_idx_arr[1]]

        ub_idx_arr = np.where(cf_found == 0)[0]
        if len(ub_idx_arr) == 1:
            logger.warning(
                f"Could not find upper bound for lambda where no solutions found. "
                f"Setting upper bound to lam_init={lams[0]}. Consider increasing lam_init to find an upper bound.")
            ub_idx = 0
        else:
            ub_idx = ub_idx_arr[-1]
        lam_ub = lams[ub_idx]

        logger.debug(f"Found upper and lower bounds for lambda: {lam_ub}, {lam_lb}")
        assert lam_lb - lam_ub > -1e-3

        # re-initialise response s.t. 'cf' and 'all' fields contain only main search resuts
        sweep_results = deepcopy(self.search_results)
        self._initialise_response()
        self.search_results['lambda_sweep']['all'] = sweep_results['all']
        self.search_results['lambda_sweep']['cf'] = sweep_results['cf']

        return {'lb': lam_lb, 'ub': lam_ub, 'midpoint': 0.5*(lam_ub + lam_lb)}

    def _cf_step(self) -> None:
        """
        Runs a gradient descent step and updates current solution.
        """

        # TODO: ALEX: TBD: GENERIC

        gradients = self._get_autodiff_gradients()
        self._apply_gradients(gradients)

    def _get_autodiff_gradients(self) -> List[tf.Tensor]:
        """
        Calculates the gradients of the loss function (specified in self.loss) with respect to the input (aka the
        counterfactual) at the current point in time.
        """

        # TODO: ALEX: SPECIFIC TO TENSORFLOW.
        # TODO: ALEX: I THINK THIS IS A VERY AWKWARD DESIGN AS IT REQUIRES THE USER TO OVERRIDE
        #  get_loss_args and get_loss_kwargs() TO ACTUALLY PASS THE ARGUMENTS TO THE METHOD. SO
        #  WE MIGHT AS WELL JUST HAVE A GENERIC IMPLEMENTATION THAT USERS CAN CUSTOMISE WITH A
        #  _SEARCH METHOD AND THEIR OWN LOSS FUNCTION. I THINK THIS IS A NOT A GOOD DESIGN.
        with tf.GradientTape() as tape:
            loss_args = self.get_loss_args()
            loss_kwargs = self.get_loss_kwargs()
            loss = self.loss(
                instance=self.instance,
                cf=self.cf,
                *loss_args,
                **loss_kwargs
            )
        gradients = tape.gradient(loss, [self.cf])

        return gradients

    def get_loss_args(self):
        # TODO: ALEX: TBD: UGLY,
        return self.lam, self.feature_scale

    def get_loss_kwargs(self):
        # TODO: ALEX: TBD: ::(
        prediction = self.get_cf_prediction(self.target_class, self.cf)

        return {'prediction': prediction, 'target_proba': self.target_proba}

    def _apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the

        """

        # TODO: ALEX: TBD: SPECIFIC TO TENSORFLOW FRAMEWORK. BLACKBOX WOULD ALSO HAVE TO OVERRIDE
        #  THIS METHOD TO INJECT THE GRADIENTS.

        autograd_grads = gradients[0]
        gradients = [self._mask * autograd_grads]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def _check_constraint(self, cf: tf.Variable, target_proba: tf.Tensor, tol: float) -> bool:
        """
        Checks if the constraint |f(cf) - target_proba| < self.tol holds where f is the model
        prediction for the class specified by the user, given the counterfactual `cf`. If the
        constraint holds, a counterfactual has been found.

        Parameters
        ----------
        cf
            Proposed counterfactual solution.
        target_proba
            Target probability for `cf`.

        Returns
        -------
        A boolean indicating whether the constraint is satisfied.

        """
        # TODO: ALEX: TBD: SPECIFIC TO THE METHOD. IN PRINCIPLE NOBODY WOULD STOP USERS WHO DEFINE
        #  THEIR SEARCH ALGORITHM TO OVERRIDE THIS METHOD SO THAT THEY CAN SPECIFY WHATEVER CONSTRAINTS THEY HAVE ON
        #  LOSS
        return tf.reduce_all(
            tf.math.abs(self.get_cf_prediction(self.target_class, cf) - target_proba) <= tol,
        ).numpy()

    def _update_search_result(self,
                              step: int,
                              lam: float,
                              current_cf: np.ndarray,
                              current_cf_pred: np.ndarray,
                              ) -> None:
        """
        Updates the model response. Called only if current solution, :math:`X'` satisfies 
        :math:`|f_t(X') - p_t| < \mathtt{tol}`. Here :math:`f_t` is the model output and `p_t` is the target model 
        output.
        
        Parameters
        ----------
        step
            Current optimisation step.
        lam
            The current value of the distance term weight.
        current_cf
            The current solution, :math:`X'`.
        current_cf_pred
            The model prediction on `current_cf`.
        
        """ # noqa W605

        # TODO: ALEX: TBD: THIS IS SPECIFIC TO THE SEARCH METHOD DUE TO LAMBDA
        # perform necessary calculations and update `self.instance_dict`
        self._collect_step_data(step, lam, current_cf, current_cf_pred)
        self.search_results['all'][self.lam_step].append(deepcopy(self.this_result))

        # update best CF if it has a smaller distance
        if not self.search_results['cf']:
            self.search_results['cf'] = deepcopy(self.this_result)
        elif self.this_result['distance_loss'] < self.search_results['cf']['distance_loss']:
            self.search_results['cf'] = deepcopy(self.this_result)

        logger.debug(f"CF found at step {step}.")

    def _collect_step_data(self, step: int, lam: float, current_cf: np.ndarray, current_cf_pred: np.ndarray):
        """
        Collects data from the current optimisation step. This data is part of the response only if the current
        optimisation step yields a solution that satisfies the constraint imposed on target output (see `explain` doc).
        Otherwise, if logging is active, the data is written to a TensorFlow event file for visualisation purposes.

        Parameters
        ----------
        See ``_update_search_result`` method.
        """

        instance = self.instance_numpy

        # compute loss terms for current counterfactual
        # TODO: ALEX: TBD: THIS IS SPECIFIC TO THE SEARCH METHOD, HAVE TO OVERRIDE.
        # TODO: ALEX: TBD: IF THE USER PROVIDES AN ARBITRARY LOSS, THIS IS NOT POSSIBLE. WE COULD GET THEM TO
        #  Return a tuple with the loss terms and a tuple with the weights and take advantage of this here...
        ax_sum = tuple(np.arange(1, len(instance.shape)))
        if self.distance_fn == 'l1':
            dist_loss = np.sum(np.abs(current_cf - instance) / self.feature_scale_numpy, axis=ax_sum)
        else:
            dist_loss = np.nan

        pred_class = self._get_label(current_cf_pred)
        pred_proba = current_cf_pred[:, pred_class]
        pred_loss = (pred_proba - self.target_proba_numpy) ** 2
        # populate the return dict
        self.this_result['X'] = current_cf
        self.this_result['lambda'] = lam
        self.this_result['step'] = step
        self.this_result['class'] = pred_class
        self.this_result['proba'] = pred_proba.item()
        self.this_result['distance_loss'] = dist_loss.item()
        self.this_result['prediction_loss'] = pred_loss.item()
        self.this_result['total_loss'] = (pred_loss + lam * dist_loss).item()

    def _write_tb(self,
                  step: int,
                  lam: float,
                  lam_lb: float,
                  lam_ub: float,
                  cf_found: np.ndarray,
                  prefix='',
                  **kwargs) -> None:
        """
        Writes data to a TensorFlow event file for visualisation purposes. Called only if the `log_optimisation_traces`
        argument to `explain` is set to `True`.

        Parameters
        ----------
        step
            The current optimisation step.
        lam
            See `explain` `lam_init` argument.
        lam_lb, lam_ub
            See `_bisect_lam`
        cf_found
            An array containing the number of counterfactuals for the current `lam` optimisation step. Shape ``(n,)``
            where `n` is the number of `lam` optimisation steps.
        prefix
            Used to create different tags so that different stages of optimisation (e.g., lambda initial search and
            main optimisation loop) are displayed separately.
        """

        # TODO: ALEX: TBD: METHOD SPECIFIC

        found = kwargs.get('found', 0)
        not_found = kwargs.get('not_found', 0)
        lr = self._get_learning_rate()
        instance_dict = self.this_result

        with tf.summary.record_if(tf.equal(step % self.summary_freq, 0)):
            with self.writer.as_default():

                # optimiser data
                tf.summary.scalar("{}global_step".format(prefix), step, step=step)
                tf.summary.scalar(
                    "{}lr/lr".format(prefix),
                    lr,
                    step=step,
                    description="Gradient descent optimiser learning rate."
                )

                # loss data
                tf.summary.scalar(
                    "{}losses/loss_dist".format(prefix),
                    instance_dict['distance_loss'],
                    step=step,
                    description="The distance between the counterfactual at the current optimisation step and the "
                                "instance to be explained."
                )
                tf.summary.scalar(
                    "{}losses/loss_pred".format(prefix),
                    instance_dict['prediction_loss'],
                    step=step,
                    description="The squared difference between the output of the model evaluated with the "
                                "the counterfactual at the current optimisation step and the model output evaluated at "
                                "the instance to be explained."
                )
                tf.summary.scalar(
                    "{}losses/total_loss".format(prefix),
                    instance_dict['total_loss'],
                    step=step,
                    description="The value of the counterfactual loss for the current optimisation step (sum loss_dist "
                                "and loss_pred"
                )
                # tf.summary.scalar("{}losses/pred_div_dist".format(prefix), loss_pred / (lam * dist), step=step)

                # distance loss weight data
                tf.summary.scalar(
                    "{}lambda/lambda".format(prefix),
                    lam,
                    step=step,
                    description="The weight of the distance_loss part of the counterfactual loss."
                )
                tf.summary.scalar(
                    "{}lambda/l_bound".format(prefix),
                    lam_lb,
                    step=step,
                    description="The upper bound of the interval on which lambda is optimised using binary search.",
                )
                tf.summary.scalar(
                    "{}lambda/u_bound".format(prefix),
                    lam_ub,
                    step=step,
                    description="The lower bound of the interval on which lambda is optimised using binary search."
                )

                tf.summary.scalar(
                    "{}lambda/lam_interval_len".format(prefix),
                    lam_ub - lam_lb,
                    step=step,
                    description="The length of the interval whithin which lambda is optimised."
                )

                # counterfactual data
                tf.summary.scalar(
                    "{}counterfactuals/prediction".format(prefix),
                    instance_dict['proba'],
                    step=step,
                    description="The output of the model at the current optimisation step.",
                )
                tf.summary.scalar(
                    "{}counterfactuals/total_counterfactuals".format(prefix),
                    cf_found.sum(),
                    step=step,
                    description="Total number of counterfactuals found in the optimisation so far."
                )
                tf.summary.scalar(
                    "{}counterfactuals/found".format(prefix),
                    found,
                    step=step,
                    description="Counter incremented when the constraint on the model output for the current "
                                "counterfactual is satisfied. Reset to 0 when the constraint is not met."
                )

                tf.summary.scalar(
                    "{}counterfactuals/not_found".format(prefix),
                    not_found,
                    step=step,
                    description="Counter incremented by one if the constraint on the model value evaluated for the "
                                "current counterfactual is not met. Resets to 0 when the constraint is satisfied",
                )

                with tf.summary.record_if(step % self.image_summary_freq == 0 and self._log_image):
                    tf.summary.image(
                        '{}counterfactuals/current_solution'.format(prefix),
                        self.cf,
                        step=step,
                        description="Current counterfactual"
                    )

        self.writer.flush()

    def _get_learning_rate(self) -> tf.Tensor:
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """

        # TODO: ALEX: TBD: TENSORFLOW SPECIFIC
        return self.optimizer._decayed_lr(tf.float32)

    def _bisect_lambda(self,
                       cf_found: np.ndarray,
                       lam_step: int,
                       lam_cf_threshold: int = 5,
                       lam_multiplier: int = 10,
                       lam_divider: int = 10) -> None:
        """
        Runs a bisection algorithm to optimise :math:`lambda`, which is adjust according to the following algorithm.
        See `explain` method documentation for details about the algorithm and parameters.
        
        Returns
        -------
        
        """ # noqa W605

        # TODO: ALEX: TBD: METHOD SPECIFC

        # lam_cf_threshold: minimum number of CF instances to warrant increasing lambda
        if cf_found[lam_step] >= lam_cf_threshold:
            self.lam_lb = max(self.lam, self.lam_lb)
            logger.debug(f"Lambda bounds: ({self.lam_lb}, {self.lam_ub})")
            if self.lam_ub < 1e9:
                self.lam = (self.lam_lb + self.lam_ub) / 2
            else:
                self.lam *= lam_multiplier
                logger.debug(f"Changed lambda to {self.lam}")

        elif cf_found[lam_step] < lam_cf_threshold:
            # if not enough solutions found so far, decrease lambda by a factor of 10,
            # otherwise bisect up to the last known successful lambda
            self.lam_ub = min(self.lam_ub, self.lam)
            logger.debug(f"Lambda bounds: ({self.lam_lb}, {self.lam_ub})")
            if self.lam_lb > 0:
                self.lam = (self.lam_lb + self.lam_ub) / 2
                logger.debug(f"Changed lambda to {self.lam}")
            else:
                self.lam /= lam_divider

    def _build_explanation(self, X: np.ndarray, result: dict, instance_class: int, instance_proba: float) -> Explanation:
        """
        Creates an explanation object and re-initialises the response to allow calling `explain` multiple times on
        the same explainer.
        """

        # TODO: ALEX: TBD: GENERIC

        # create explanation object
        result['instance'] = X
        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=result)
        # reset response
        self._initialise_response()

        return explanation

    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        This function updates the metadata of the explainer using the data from
        the `data_dict`. If the params option is specified, then each key-value
        pair is added to the metadata `'params'` dictionary only if the key is
        included in `CF_PARAMS`.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
        """

        # TODO: ALEX: TBD: GENERIC

        if params:
            for key in data_dict.keys():
                if key not in CF_PARAMS:
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)


# TODO: ALEX: Test that the constrains are appropriate when calling fit with a dataset
1import copy
import functools
import logging
import os

import numpy as np
import tensorflow as tf

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.utils.gradients import numerical_gradients
from collections import defaultdict
from copy import deepcopy
from functools import partial
from inspect import signature
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _wrap_black_box_predictor_tensorflow(func):
    """
    A decorator that converts the first arguemnt to `func` to a numpy array and converts the output to a tensor.
    """
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        X, *others = args
        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            X = X.numpy()
        result = func(X, *others, **kwargs)
        return tf.identity(result)
    return wrap


def _wrap_black_box_predictor_pytorch(func):
    raise NotImplementedError("PyTorch is not suported at the moment!")

# TODO: ALEX: TBD I think it's best practice to define all the custom
#  errors in one file (e.g., exceptions.py/error.py). We
#  could standardise this across the library


class CounterfactualError(Exception):
    pass


def scaled_l1_loss(instance: tf.Tensor, cf: tf.Variable, feature_scale: Optional[tf.Tensor] = None) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    # TODO: ALEX: TBD: USE TF BUILTINS

    ax_sum = tuple(np.arange(1, len(instance.shape)))
    if feature_scale is None:
        return tf.reduce_sum(tf.abs(cf - instance), axis=ax_sum, name='l1')
    return tf.reduce_sum(tf.abs(cf - instance) / feature_scale, axis=ax_sum, name='l1')

# TODO: WHY IS PREDICTED PROBA TYPED AS VARIABLE?


def squared_loss(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    # TODO: ALEX: TBD: THERE SHOULD BE A KERAS BUILT IN ?
    return tf.square(pred_probas - target_probas)

# TODO: WHY IS PREDICTED PROBA TYPED AS VARIABLE?


def squared_loss_grad(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    return 2*(pred_probas - target_probas)


CF_VALID_SCALING = ['MAD']
CF_PARAMS = ['scale_loss', 'constrain_features', 'feature_whitelist']

WATCHER_LAM_OPTS_DEFAULT = {
    'lam_init': 0.1,
    'lams': None,  # TODO: Mention this is should be sorted high to low
    'nb_lams': 2,  # TODO: DOCUMENT THIS PARAMETER
    'lam_exploration_steps': 20,  # TODO: DOCUMENT THIS PARAM
    'instance_proba_delta': 0.001, # TODO: DOCUMENT THIS PARAM
     # the model output evaluated at the original instance should decrease by at least this value in lam_exploration_steps to warrant continuation of sweep
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
To override the `max_lam_steps` parameters at explain time, call `explain` with they keyword argument::

    lam_opts = {'max_lam_steps': 50}.
""" # noqa W605


WATCHER_SEARCH_OPTS_DEFAULT = {
    'max_iter': 1000,
    'early_stop': 50,
}
"""dict: The default values governing the search process. See `explain` method for a detailed descriptions of each
parameter.

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys to the
explainer constructor or when calling `explain`. If the same subset of arguments is specified in both `explain` and the
constructor, the `explain` options will override the constructor options.

Examples
--------
To override the `early_stop` parameter at explain time, call `explain` with the keyword argument::

    search_opts = {'early_stop': 100}.
"""
CF_LOGGING_OPTS_DEFAULT = {
    'verbose': False,
    'log_traces': True,
    'trace_dir': 'logs/cf',
    'summary_freq': 1,
    'image_summary_freq': 10,
}
"""
dict: The default values for logging options. See `explain` method for a detailed descriptions of each parameter.

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys when calling
`explain`

Examples
--------
To specify a the name logging directory for TensorFlow event files and a new frequency for logging images call `explain`
with the following key-word arguments::

    logging_opts = {'trace_dir': 'experiments/cf', 'image_summary_freq': 5}
"""

# TODO: ADD DOCSTRING
WATCHER_METHOD_OPTS = {
    'tol': 0.01,
    'search_opts': WATCHER_SEARCH_OPTS_DEFAULT,
    'lam_opts': WATCHER_LAM_OPTS_DEFAULT,
}
# TODO: ADD DOCSTRING
WATCHER_LOSS_SPEC_WHITEBOX = {
    'prediction': {'fcn': squared_loss, 'kwargs': {}},
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
}
# TODO: ADD DOCSTRING
WATCHER_LOSS_SPEC_BLACKBOX = {
    'prediction': {
        'fcn': squared_loss,
        'kwargs': {},
        'grad_fcn': squared_loss_grad,
        'grad_fcn_kwargs': {},
        'gradient_method': {'name': 'num_grad_batch_tensorflow', 'kwargs': {}}
        # TODO: ALEX: TBD: WE COULD INLUDE THE ACTUAL FUNCTION HERE, BUT I AM NOT SURE I SEE THE POINT?
        # 'watcher_black_box_wrapper': None
    },
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'eps': 0.01
}
loss_terms = [f"{term}_fcn" for term in WATCHER_LOSS_SPEC_WHITEBOX.keys()]

CF_ATTRIBUTES = set(WATCHER_SEARCH_OPTS_DEFAULT.keys()) | \
                set(WATCHER_LAM_OPTS_DEFAULT.keys()) | \
                set(CF_LOGGING_OPTS_DEFAULT) |\
                set(loss_terms) |\
                {'tol', 'prediction_grad_fcn', '_num_grads_fn'}

NUMERICAL_GRADIENT_FUNCTIONS = [fcn.__name__ for fcn in numerical_gradients]


def watcher_loss(distance: tf.Tensor, lam: float, pred: tf.Tensor) -> tf.Tensor:

    # TODO: DOCSTRING

    return lam * distance + pred


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


def _convert_to_label(Y: np.ndarray, threshold: float = 0.5) -> int:
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

    # TODO: ALEX: TBD: Parametrise `threshold` in explain or do we assume people always assign labels on this rule?

    if Y.shape[1] == 1:
        return int(Y > threshold)
    else:
        return np.argmax(Y)


def _slice_prediction(prediction: tf.Tensor, target_idx: Union[int, str], src_idx: int) -> tf.Tensor:
    """
    Returns a slice from `prediction` depending on the value of the `target_idx`.

    Parameters
    ----------
    prediction
        A prediction tensor
    target_idx

        - If ``int``, the slice of `prediction` indicated by it is returned
        - If 'same', the slice indicated by `scr_idx` is returned
        - If 'other', the slice that maximises the prediction other than `src_idx` is returned

    src_idx
        An index in the prediction that influences the returned slice as described above.
    """

    if isinstance(target_idx, int) or target_idx == 'same':
        return prediction[:, target_idx]
    if target_idx != 'other':
        raise ValueError("The only allowed values for output slices are {'same','other'}")

    _, indices = tf.math.top_k(prediction, k=2)
    if indices[0][0] == src_idx:
        cf_prediction = prediction[:, indices[0][1].numpy()]
    else:
        cf_prediction = prediction[:, indices[0][0].numpy()]
    return cf_prediction


def watcher_blackbox_wrapper(X: tf.Tensor,
                             predictor: Callable,
                             target_idx: Union[str, int],
                             src_idx: int) -> tf.Tensor:

    # TODO: DOCSTRING
    # TODO: WHEN WRITING DOCS, SPECIFY THE SHAPE THAT THE PREDICTOR IS EXPECTED TO TAKE IN AND RETURN
    prediction = predictor(X)
    pred_target = _slice_prediction(prediction, target_idx, src_idx)
    return pred_target


# TODO: ENSURE VARIABLES ARE INITIALISED BEFORE CALLING SEARCH
# TODO: IMPLEMENT SCHEDULER FOR PYTORCH
# TODO: SETATTRIBUTE SHOULD TAKE AN ITERABLE OF KEY-VALUE PAIRS ALSO

class WatcherCounterfactualBase:

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: Tuple[int, ...],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
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

        # TODO: ALEX: UPDATE DOCSTRINGS
        # TODO: ALEX: IN WRAPPER CLASS, UPDATE METADATA
         # TODO: ALEX: THINK OF HOW CONSTRAINTS COULD BE SPECIFIED TO THE WRAPPER

        self.fitted = False

        self.predictor = predictor
        if shape[0] != 1:
            logger.warning(
                f"shape argument passed to the constructor was {shape[0]} but expected value is 1. Batch explanation "
                f"is not supported by this explainer! Ensure only one example is passed at each explain call.",
            )

        # create attribute and set them with default values
        self._set_attributes(WATCHER_METHOD_OPTS['search_opts'])
        self._set_attributes(WATCHER_METHOD_OPTS['lam_opts'])
        self._set_attributes({'tol': WATCHER_METHOD_OPTS['tol']})
        # override defaults with user specification
        if method_opts:
            for key in method_opts:
                if isinstance(method_opts[key], dict):
                    self._set_attributes(method_opts[key])
                else:
                    self._set_attributes({key: method_opts[key]})
            # TODO: ADD SOME CHECKING FOR UNEXPECTED ATTRIBUTES TO HELP CATCHING TYPOS IN METHOD_OPTS EASILY

        # set default logging options
        self._set_attributes(CF_LOGGING_OPTS_DEFAULT)
        # check shape of data for logging purposes
        self._log_image = False
        if len(shape) == 4:
            self._log_image = True

        # set all the functions in the
        if loss_spec is None:
            if predictor_type == 'blackbox':
                loss_spec = WATCHER_LOSS_SPEC_BLACKBOX
            else:
                loss_spec = WATCHER_LOSS_SPEC_WHITEBOX
        for term in loss_spec:
            # defer this to black-box subclasses
            # TODO: THIS IS NO LONGER NECESSARY I GUESS
            if 'grad_fn' in loss_spec[term]:
                continue
            this_term_kwargs = loss_spec[term]['kwargs']
            if this_term_kwargs:
                this_term_fcn = partial(loss_spec[term]['fcn'], **this_term_kwargs)
                self._set_attributes({f"{term}_fcn": this_term_fcn})
            else:
                self._set_attributes({f"{term}_fcn": loss_spec[term]['fcn']})
        self.loss_spec = loss_spec

        self._current_loss = dict.fromkeys(loss_spec.keys(), 0.0)
        self._current_loss.update([('combined', 0.0)])
        self.loss_fcn = watcher_loss

        self.optimizer = None
        self._optimizer_copy = None

        # TODO: ADD TYPES FOR OTHER FRAMEWORKS AS THEY GET IMPLEMENTED
        # init. at explain time
        self.target_proba = None  # type: Union[tf.Tensor, None]
        self.instance = None  # type: Union[tf.Tensor, None]
        self.cf = None  # type: Union[tf.Variable, None]

        # initialisation method and constraints for counterfactual
        self.cf_constraint = None
        if feature_range is not None:
            self.cf_constraint = partial(range_constraint, low=feature_range[0], high=feature_range[1])

        # scheduler and optimizer initialised at explain time
        self.step = -1
        self.lam_step = -1

        # return templates
        self._initialise_response()
        # nb. calls to the counterfactual method
        self._num_calls = 0
        # subclasses provide PyTorch support, set by wrapper
        self.device = None

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)

    def __getattr__(self, key: Any) -> Any:
        return self.__getattribute__(key)

    def _set_attributes(self, attrs: Union[Dict[str, Any], None]) -> None:
        """
        Sets the attributes of the explainer using the (key, value) pairs in attributes. Ignores attributes that
        are not specified in CF_ATTRIBUTES.
        """

        # called with None if the attributes are not overridden
        if not attrs:
            return

        for key, value in attrs.items():
            if key not in CF_ATTRIBUTES:
                logger.warning(f"Attribute {key} unknown. Ignoring ...")
                continue
            self.__setattr__(key, value)

    def _initialise_response(self) -> None:
        """
        Initialises the templates that will form the body of the `explanation.data` field.
        """

        # NB: (class, proba) could be renamed to (output_idx, output) to make sense for regression settings
        self.this_result = dict.fromkeys(
            [
                'X',
                'distance_loss',
                'prediction_loss',
                'total_loss',
                'lambda',
                'step',
                'target_class_proba',
                'target_class',
                'original_class_proba'
            ]
        )
        self.search_results = copy.deepcopy(DEFAULT_DATA_CF)
        self.search_results['all'] = defaultdict(list)
        self.search_results['lambda_sweep'] = {}

    def _search(self, *, init_cf: np.ndarray) -> dict:
        """
        Searches a counterfactual given an initial condition for the counterfactual. The search has two loops:

            - An outer loop, where :math:`\lambda` (the weight of the distance between the current counterfactual and \
            the input `X` to explain in the loss term) is optimised using bisection

            - An inner loop, where for constant `lambda` the current counterfactual is updated using the gradient of \
            the counterfactual loss function.

        Parameters
        ----------
        init_cf
            Initial condition for the optimisation.

        Returns
        -------
        A dictionary containing the search results, as defined in `alibi.api.defaults`.
        """  # noqa: W605


        # find a range to optimise lambda in and an initial value
        lam_dict = self._initialise_lam(
            self.instance_proba_delta,
            self.nb_lams,
            self.lam_exploration_steps,
            lams=self.lams,
            decay_steps=self.decay_steps,
            common_ratio=self.common_ratio,
        )
        init_lam,  init_lb, init_ub = lam_dict['midpoint'], lam_dict['lb'], lam_dict['ub']
        self._bisect_lambda = partial(
            self._bisect_lam,
            lam_cf_threshold=self.lam_cf_threshold,
            lam_multiplier=self.lam_multiplier,
            lam_divider=self.lam_divider,
        )

        summary_freq = self.summary_freq
        cf_found = np.zeros((self.max_lam_steps, ))
        # re-init. cf as initial lambda sweep changed the initial condition
        self._initialise_cf(init_cf)
        lam, lam_lb, lam_ub = init_lam, init_lb, init_ub

        for lam_step in range(self.max_lam_steps):
            self.lam = lam
            self.lam_step +=1
            # re-set learning rate
            self._reset_optimizer()
            found, not_found = 0, 0
            for gd_step in range(self.max_iter):
                self._cf_step()
                self.step += 1
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self._make_prediction(self.cf)

                # save and optionally display results of current gradient descent step
                current_state = (self.step, self.to_numpy_arr(self.cf), self.to_numpy_arr(cf_prediction))
                write_summary = self.log_traces and self.step % summary_freq == 0
                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._record_loss()
                    self._update_response(*current_state)
                    found += 1
                    not_found = 0
                else:
                    found = 0
                    not_found += 1
                    if write_summary:
                        self._record_loss()
                        self._collect_step_data(*current_state)

                if write_summary:
                    self._record(self.step, self.lam, lam_lb, lam_ub, cf_found, prefix='counterfactual_search/')

                # early stopping criterion - if no solutions or enough solutions found, change lambda
                if found >= self.early_stop or not_found >= self.early_stop:
                    break

            lam, lam_lb, lam_ub = self._bisect_lambda(
                cf_found,
                lam_step,
                self.lam,
                lam_lb,
                lam_ub,
            )

        self._display_solution(prefix='best_solutions/')

        return deepcopy(self.search_results)

    def _initialise_lam(self,
                        instance_proba_delta: float,
                        nb_lams: int,
                        lam_exploration_steps: int,
                        lams: Optional[np.ndarray] = None,
                        decay_steps: int = 10,
                        common_ratio: int = 10) -> Dict[str, float]:
        """
        Runs a search procedure over a specified range of lambdas in order to determine a range of values of this
        parameter for which counterfactuals exist. If `lams` is not specified, a range of lambda is created by diving an
        initial value by the geometric sequence ``[decay_factor**i for i in range(decay_steps)]``. The method saves the results of this sweep in the
        `'lambda_sweep'` field of the explanation 'data' field.

        Parameters
        ----------
        instance_proba_delta
            If the absolute difference between the model output on the instance whose counterfactual is searched and the
            equivalent prediction on the current solution is less than this tolerance, the probability predicted is
            considered constant.
        lam_exploration_steps
            If the probability of the instance class remains constant for `lam_exploration_steps`, the optimisation
            is ran with a new `lam` value.
        nb_lams:
            The number of distinct values of `lam` for which counterfactuals are found that need to be evaluated before
            terminating the initialisation.
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

        n_steps = self.max_iter // decay_steps
        if lams is None:
            lams = np.array([self.lam_init / common_ratio ** i for i in range(decay_steps)])  # exponential decay
        # in-place sort
        lams[::-1].sort()
        cf_found = np.zeros(lams.shape, dtype=np.uint16)

        for lam_step, lam in enumerate(lams):
            explored_steps = 0
            # optimiser is re-created so that lr schedule is reset for every lam
            self._reset_optimizer()
            self.lam = lam
            self.lam_step += 1
            for gd_step in range(n_steps):
                # update cf with loss gradient for a fixed lambda
                self._cf_step()
                self.step += 1
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self._make_prediction(self.cf)

                # save search results and log to TensorBoard
                write_summary = self.log_traces and self.step % self.summary_freq == 0
                current_state = (self.step, self.to_numpy_arr(self.cf), self.to_numpy_arr(cf_prediction))

                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._record_loss()
                    self._update_response(*current_state)
                    if write_summary:
                        self._record(self.step, self.lam, 0, 0, cf_found, prefix='lambda_sweep/')
                    # sufficient to find a counterfactual for a given lambda in order to consider that lambda valid
                    break
                else:
                    if write_summary:
                        self._record_loss()
                        self._collect_step_data(*current_state)
                        self._record(self.step, self.lam, 0, 0, cf_found, prefix='lambda_sweep/')
                    instance_class_pred = self.to_numpy_arr(cf_prediction[:, self.instance_class]).item()
                    # TODO: THIS SHOULD BE MORE SUBTLE AND ACCOUNT FOR THE CASE WHEN `TARGET CLASS = 'SAME'` AND
                    #  WHETHER THE SEARCH SHOULD INCREASE/DECREASE THE PREDICTION PROBABILITY. IF CLASS IS 'OTHER'
                    #  OR INT DIFFERENT TO THE PRED CLASS, THEN WE SHOULD INCREMENT THE EXPLORED_STEPS IF THE PROBA
                    #  OF THE ORIGINAL CLASS INCREASES BY MORE THEN INSTANCE_PROBA_DELTA
                    if abs(self.instance_proba - instance_class_pred) < instance_proba_delta:
                        explored_steps += 1
                        # stop optimising at the current lambda value if the probability of the original class
                        # does not change significantly
                        if explored_steps == lam_exploration_steps:
                            break
                # finialise initialisation if counterfactuals were found for `nb_lams` lambda values. The extremes
                # found will constitute the interval for optimising lambda
            if cf_found.sum() == nb_lams:
                break

        self._reset_step()
        logger.debug(f"Counterfactuals found: {cf_found}")

        # determine upper and lower bounds given the solutions found
        # TODO: REPLACE WITH NAMEDTUPLE
        lambda_dict = {'lb': 0.0, 'ub': 0.0, 'midpoint': 0.0}
        if cf_found.sum() == 0:
            raise CounterfactualError(
                f"No counterfactual was found for any lambda among {lams}. If the current settings for the target "
                f"class and tolerance are strict, then increase lam_exploration steps to allow for a longer exploration"
                f"phase at each value of lambda and/or reduce 'instance_delta_proba', the threshold over which a "
                f"deviation from the predicted probability on the original instance is considered significant. "
                f"Otherwise, adjusting the target_proba, target_class and tol may help to find counterfactuals."
            )
        elif cf_found.sum() == len(cf_found):
            # found a cf for all lambdas
            lambda_dict['ub'] = lams[0]
            lambda_dict['lb'] = lams[1]
        elif cf_found.sum() == 1:
            # this case is unlikely to occur in practice
            if cf_found[0] == 1:
                lambda_dict['lb'] = lams[0]
                lambda_dict['ub'] = 10*lambda_dict['lb']
            else:
                valid_lam_idx = np.nonzero(cf_found)[0][0]
                lambda_dict['lb'] = lams[valid_lam_idx]
                lambda_dict['ub'] = lams[valid_lam_idx - 1]
        else:
            # assumes that as lambda decreases counterfactuals are found
            lb_idx_arr = np.where(cf_found > 0)[0]
            lam_lb_idx = lb_idx_arr[1]  # second largest lambda for which cfs are found
            # backtrack to find the upper bound
            lam_ub_idx = lam_lb_idx - 1
            while lam_ub_idx >= 0:
                if cf_found[lam_lb_idx] > 0:
                    break
                lam_ub_idx -= 1
            lambda_dict['lb'] = lams[lam_lb_idx]
            lambda_dict['ub'] = lams[lam_ub_idx]
        lambda_dict['midpoint'] = 0.5*(lambda_dict['lb'] + lambda_dict['ub'])

        logger.debug(f"Found upper and lower bounds for lambda: {lambda_dict['ub']}, {lambda_dict['lb']}")

        # re-initialise response s.t. 'cf' and 'all' fields contain only main search results
        sweep_results = deepcopy(self.search_results)
        self._initialise_response()
        self.search_results['lambda_sweep']['all'] = sweep_results['all']
        self.search_results['lambda_sweep']['cf'] = sweep_results['cf']

        return lambda_dict

    def to_numpy_arr(self, X: Union[tf.Tensor, tf.Variable, np.ndarray], **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclass must implement conversion to numpy array!")

    def to_tensor(self, X: np.ndarray, **kwargs) -> tf.Tensor:
        raise NotImplementedError("Subclass must implement convertion to tensor object!")

    def _reset_optimizer(self):
        raise NotImplementedError("Subclass must implement optimizer reset method !")

    def _set_optimizer(self, optimizer: Union[tf.keras.optimizers.Optimizer], optimizer_opts: Optional[Dict]):
        raise NotImplementedError("Subclass must implement optimizer set method!")

    def _cf_step(self):
        raise NotImplementedError("Subclass must implement counterfactual update method!")

    def _initialise_cf(self, init_cf: np.ndarray) -> None:
        raise NotImplementedError("Subclass must implement counterfactual initialisation method!!")

    # TODO: ALEX: ADD TYPES
    def _check_constraint(self, counterfactual, target, tolerance):
        raise NotImplementedError("Subclass must implement model output constraint method!")

    # TODO: ALEX: ADD TYPES
    def _make_prediction(self, X):
        raise NotImplementedError("Subclass must implement predictor call method!")

    @staticmethod
    def _bisect_lam(cf_found: np.ndarray,
                    lam_step: int,
                    lam: float,
                    lam_lb: float,
                    lam_ub: float,
                    lam_cf_threshold: int = 5,
                    lam_multiplier: int = 10,
                    lam_divider: int = 10) -> Tuple[float, float, float]:
        """
        Runs a bisection algorithm to optimise :math:`lambda`, which is adjust according to the following algorithm.
        See `explain` method documentation for details about the algorithm and parameters.

        Returns
        -------

        """  # noqa W605

        # lam_cf_threshold: minimum number of CF instances to warrant increasing lambda
        if cf_found[lam_step] >= lam_cf_threshold:
            lam_lb = max(lam, lam_lb)
            logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_ub < 1e9:
                lam = (lam_lb + lam_ub) / 2
            else:
                lam *= lam_multiplier
                logger.debug(f"Changed lambda to {lam}")

        elif cf_found[lam_step] < lam_cf_threshold:
            # if not enough solutions found so far, decrease lambda by a factor of 10,
            # otherwise bisect up to the last known successful lambda
            lam_ub = min(lam_ub, lam)
            logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_lb > 0:
                lam = (lam_lb + lam_ub) / 2
                logger.debug(f"Changed lambda to {lam}")
            else:
                lam /= lam_divider

        return lam, lam_lb, lam_ub

    def _collect_step_data(self, step: int, current_cf: np.ndarray, current_cf_pred: np.ndarray):
        """
        Collects data from the current optimisation step. This data is part of the response only if the current
        optimisation step yields a solution that satisfies the constraint imposed on target output (see `explain` doc).
        Otherwise, if logging is active, the data is written to a TensorFlow event file for visualisation purposes.

        Parameters
        ----------
        See ``_update_search_result`` method.
        """

        pred_class = _convert_to_label(current_cf_pred)
        target_pred_proba = current_cf_pred[:, pred_class]
        instance_class_proba = current_cf_pred[:, self.instance_class]
        self.this_result['X'] = current_cf
        self.this_result['lambda'] = self.lam
        self.this_result['step'] = step
        self.this_result['target_class'] = pred_class
        self.this_result['target_class_proba'] = target_pred_proba.item()
        self.this_result['instance_class_proba'] = instance_class_proba.item()

    def _record_loss(self):

        # TODO: DOCSTRING
        for key in self.loss_spec:
            self.this_result[f'{key}_loss'] = self._current_loss[key]
        self.this_result['total_loss'] = self._current_loss['combined']

    def _update_response(self,
                         step: int,
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
        current_cf
            The current solution, :math:`X'`.
        current_cf_pred
            The model prediction on `current_cf`.

        """  # noqa W605

        # TODO: CAN THIS BE IMPROVED? THIS METHOD SHOULD ARGUABLY JUST READ INFO FROM THE CURRENT_STEP DATA AND ADD IT
        #  TO THE RESPONSE. IT MAY ADD ADDITIONAL INFO APART FROM THAT AS NECESSARY BUT IT DOES NOT NEED TO PUT
        #  ALL THE STUFF IN CURRENT_STEP DATA in _search_result. THE COLLECT_STEP_DATA FUNCTION SHOULD JUST RECORD
        #  ALL RELEVANT STATE FOR THE PURPOSES OF LOGGING

        # perform necessary calculations and update `self.instance_dict`
        self._collect_step_data(step, current_cf, current_cf_pred)
        self.search_results['all'][self.lam_step].append(deepcopy(self.this_result))

        # update best CF if it has a smaller distance
        if not self.search_results['cf']:
            self.search_results['cf'] = deepcopy(self.this_result)
        elif self.this_result['distance_loss'] < self.search_results['cf']['distance_loss']:
            self.search_results['cf'] = deepcopy(self.this_result)

        logger.debug(f"CF found at step {step}.")

    def _record(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement trace logging method!")

    def _setup_tb(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement TensorBoard setup!")

    def _display_solution(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement image logging method!")

    def _reset_step(self):
        """
        Resets the optimisation step for gradient descent and for the weight optimisation step (`lam_step`).
        """

        self.step = -1
        self.lam_step = -1


class WatcherTFHelper(WatcherCounterfactualBase):

    # TODO: ALEX: TBD: CHECK TYPING FOR PREDICTOR
    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: Tuple[int, ...],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[Dict] = None,
                 method_opts: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        super().__init__(predictor, shape, predictor_type, loss_spec, method_opts, feature_range, **kwargs)
        self._cf_prediction = _slice_prediction

        # below would need to be done for a PyTorchHelper with GPU support
        # self.to_numpy_arr = partial(self.to_numpy_arr, device=self.device)

    def _set_default_optimizer(self) -> None:
        """
        Initialises the explainer with a default optimizer.
        """
        # TODO: ENSURE THIS IS DOCUMENTED
        self.lr_schedule = PolynomialDecay(0.1, self.max_iter, end_learning_rate=0.002, power=1)
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    def _reset_optimizer(self) -> None:
        """
        Resets the default optimizer. It is used to set the optimizer specified by the user as well as reset the 
        optimizer at each :math:`\lambda` optimisation cycle.

        Parametrs
        ---------
        optimizer
            This is either the opimizer passed by the user (first call) or a copy of it (during `_search`).
        """  # noqa W605

        self.optimizer = deepcopy(self._optimizer_copy)

    def _set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer, optimizer_opts: Dict[str, Any]) -> None:

        # TODO: DOCSTRING
        # TODO: TBD: Technically, _set_default_optimizer is redundant and we could have a default spec for the optimizer
        #  and call this method to set the default. This comes with the caveat that the user will have to also
        #  remember to reset the optimizer if they re-set max_iter and want to use default, which might be annoying.

        # create a backup if the user does not override
        if optimizer is None:
            self._set_default_optimizer()
            self._optimizer_copy = deepcopy(self.optimizer)
            return

        # user passed the initialised object
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            self.optimizer = optimizer
        # user passed just the name of the class
        else:
            if optimizer_opts is not None:
                self.optimizer = optimizer(**optimizer_opts)
            else:
                self.optimizer = optimizer()

        self._optimizer_copy = deepcopy(self.optimizer)

    def _initialise_variables(self, X: np.ndarray, optimised_features: np.ndarray, target_class: Union[int, str],
                              instance_class: int, instance_proba: float, target_proba: float) -> None:
        """
        Initialises optimisation variables so that the TensorFlow auto-differentiation framework
        can be used for counterfactual search.
        """

        # TODO: ALEX: DOCSTRING

        # tf.identity is the same as constant but does not always create tensors on CPU
        self.target_proba = tf.identity(target_proba * np.ones(1, dtype=X.dtype), name='target_proba')
        self.target_class = target_class
        self.instance = tf.identity(X, name='instance')
        self.instance_class = instance_class
        self.instance_proba = instance_proba
        self._initialise_cf(X)
        self._mask = tf.identity(optimised_features, name='gradient mask')

    def _cf_step(self) -> None:
        """
        Runs a gradient descent step and updates current solution.
        """

        gradients = self._get_autodiff_gradients()
        self._apply_gradients(gradients)

    def _get_autodiff_gradients(self) -> List[tf.Tensor]:
        """
        Calculates the gradients of the loss function with respect to the input (aka the counterfactual) at the current
        iteration.
        """

        with tf.GradientTape() as tape:
            loss = self._autograd_loss()
        gradients = tape.gradient(loss, [self.cf])

        return gradients

    def _setup_tb(self):
        """
        Creates a summary file writer for the current explain call. A new event file is created for each
        call to the `counterfactual` method.
        """

        # TODO: DOCSTRING
        if self.log_traces:
            trace_dir = os.path.join(self.trace_dir, f"run_{self._num_calls + 1}")
            self.writer = tf.summary.create_file_writer(trace_dir)
        self._num_calls += 1

    def _record(self,
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

        # TODO: ALEX: THIS METHOD SHOULD NOT BE PART OF THIS OBJECT. CREATE A RECORDER OBJECT THAT KNOWS IN ADVANCE
        #  WHAT IS TO BE RECORED. AT THE SAME TIME CREATE A NAMEDTUPLE FROM THAT RECORDER THAT THE METHODS CAN UPDATE
        #  AS THEY LIKE. AT EVERY STEP, THE RECORDER GETS THE NAMED TUPLE AND USES ITS INTERNAL LOGIC TO WRITE THE DATA
        #  TO THE EVENT FILE. THERE SHOULD BE ONE RECORDER FOR EACH FRAMEWORK. EACH ALGORITHM HAS A CONFIGURATION WITH
        #  WHAT IS RECORDED BY DEFAULT AND WHAT NOT. HARMONIZE THIS WITH _COLLECT_STEP_DATA/_UPDATE_RESPONSE METHODS

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
                    "{}counterfactuals/target_class_proba".format(prefix),
                    instance_dict['target_class_proba'],
                    step=step,
                    description="Predicted probability for the target class.",
                )
                tf.summary.scalar(
                    "{}counterfactuals/target_class".format(prefix),
                    instance_dict['target_class'],
                    step=step,
                    description="Predicted probability for the target class",
                )

                tf.summary.scalar(
                    "{}counterfactuals/instance_class_proba".format(prefix),
                    instance_dict['instance_class_proba'],
                    step=step,
                    description="Probability associated with the predicted instance label for the current solution",
                )

                tf.summary.scalar(
                    "{}counterfactuals/total_counterfactuals".format(prefix),
                    cf_found.sum(),
                    step=step,
                    description="Total number of counterfactuals found in the optimisation so far."
                )

                with tf.summary.record_if(step % self.image_summary_freq == 0 and self._log_image):
                    tf.summary.image(
                        '{}counterfactuals/current_solution'.format(prefix),
                        self.cf,
                        step=step,
                        description=f"Current counterfactual (lam = {self.lam})"
                    )

        self.writer.flush()

    def _make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Makes a prediction for data points in `X`.

        Parameters
        ----------
        X
            A tensor or array for which predictions are requested.
        """

        return self.predictor(X, training=False)

    def _display_solution(self, prefix: str = '') -> None:
        """
        Displays the instance along with the counterfactual with the smallest distance from the instance.

        Parameters
        ----------
        prefix
            A string `used to place the solution into a separate Tensorboard tab.
        """

        # TODO: THIS METHOD SHOULD BE PART OF THE RECORDER CLASS
        if not self.log_traces:
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

    def _initialise_cf(self, X: np.ndarray) -> None:
        """
        Initializes the counterfactual to the data point `X`.

        Parameters
        ----------
        X
            Instance whose counterfactual is to be found.
        """

        # TODO: UPDATE DOCSTRING

        self.cf = tf.Variable(
            initial_value=X,
            trainable=True,
            name='counterfactual',
            constraint=self.cf_constraint,
        )

    def _get_learning_rate(self) -> tf.Tensor:
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """
        return self.optimizer._decayed_lr(tf.float32)

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

        prediction = self._make_prediction(cf)
        return tf.reduce_all(
            tf.math.abs(self._get_cf_prediction(prediction, self.target_class) - target_proba) <= tol,
        ).numpy()

    def to_numpy_arr(self, X: Union[tf.Tensor, tf.Variable, np.ndarray], **kwargs) -> np.ndarray:
        """
        Casts an array-like object tf.Tensor and tf.Variable objects to a `np.array` object.
        """

        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            return X.numpy()
        return X

    def to_tensor(self, X: np.ndarray, **kwargs) -> tf.Tensor:
        """
        Casts a numpy array object to tf.Tensor.
        """
        return tf.identity(X)

    def copy(self, X: tf.Variable, **kwargs) -> tf.Tensor:
        """
        Copies the value of the variable X into a new tensor
        """
        return tf.identity(X)


class WatcherCFSearchWhiteBoxTF(WatcherTFHelper):

    # TODO: ALEX: TBD: CHECK TYPING FOR PREDICTOR
    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: Tuple[int, ...],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[Dict] = None,
                 method_opts: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        super().__init__(predictor, shape, predictor_type, loss_spec, method_opts, feature_range, **kwargs)

    def _autograd_loss(self) -> tf.Tensor:
        """
        Computes:
            - the prediction of the model to be explained given a target class and a counterfactual
            - the loss of the model prediction given a target prediction
            - the distance between the instance whose counterfactual is searched and the current solution.

        Returns
        -------
        A scalar representing the combined prediction and distance losses according to the function specified by the
        `loss_fcn` attribute.
        """

        # TODO: DOCSTRING

        dist_loss = self.distance_fcn(self.instance, self.cf)
        model_output = self._make_prediction(self.cf)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        combined_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self._current_loss['distance'] = self.to_numpy_arr(dist_loss).item()
        self._current_loss['prediction'] = self.to_numpy_arr(pred_loss).item()
        self._current_loss['combined'] = self.to_numpy_arr(combined_loss).item()

        return combined_loss

    def _apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the gradients of the differentiable part of the loss.
        """

        # TODO: THIS METHOD MAKES THE IMPLEMENTATION OF THIS CLASS FRAMEWORK SPECIFIC? IT'S ANNOYING BECAUSE MOST
        #  OF THE CODE IS GOING TO BE EXACTLY THE SAME. CAN WE DO BETTER?

        autograd_grads = gradients[0]
        gradients = [self._mask * autograd_grads]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def counterfactual(self,
                       instance: np.ndarray,
                       instance_class: int,
                       instance_proba: float,
                       optimised_features: np.ndarray,
                       target_class: Union[str, int] = 'other',
                       target_proba: float = 1.0,
                       optimizer:Optional[tf.keras.optimizers.Optimizer] = None,
                       optimizer_opts: Optional[Dict] = None,
                       ):
        """
        Parameters
        ----------
        optimised_features
            Binary mask. A 1 indicates that the corresponding feature can be changed during search.
        """

        # TODO: DOCSTRING

        # check inputs
        if instance.shape[0] != 1:
            raise CounterfactualError(
               f"Only single instance explanations supported (leading dim = 1). Got leading dim = {instance.shape[0]}",
            )

        self._setup_tb()
        self._get_cf_prediction = partial(self._cf_prediction, src_idx=instance_class)
        self._initialise_variables(instance, optimised_features, target_class, instance_class, instance_proba, target_proba)
        self._set_optimizer(optimizer, optimizer_opts)
        result = self._search(init_cf=instance)
        self._reset_step()
        self._reset_optimizer()

        return result


class WatcherCFSearchBlackBoxTF(WatcherCFSearchWhiteBoxTF):

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: Tuple[int, ...],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[Dict] = None,
                 method_opts: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        super().__init__(predictor, shape, predictor_type, loss_spec, method_opts, feature_range, **kwargs)

        # set properties for the non-differentiable loss terms
        # use self.loss_spec as loss_spec=None is handled in superclass
        self._num_grads_fn = None
        for term in self.loss_spec:
            if 'grad_fcn' in self.loss_spec[term]:
                this_term_kwargs = self.loss_spec[term]['kwargs']
                if this_term_kwargs:
                    this_term_fcn = partial(self.loss_spec[term]['fcn'], **this_term_kwargs)
                    self._set_attributes({f"{term}_fcn": this_term_fcn})
                else:
                    self._set_attributes({f"{term}_fcn": self.loss_spec[term]['fcn']})
                this_term_grad_fcn_kwargs = self.loss_spec[term]['grad_fcn_kwargs']
                if this_term_grad_fcn_kwargs:
                    this_term_grad_fcn = partial(self.loss_spec[term]['grad_fcn'])
                    self._set_attributes({f"{term}_grad_fcn": this_term_grad_fcn})
                else:
                    self._set_attributes({f"{term}_grad_fcn": self.loss_spec[term]['grad_fcn']})
                grad_method_idx = NUMERICAL_GRADIENT_FUNCTIONS.index(self.loss_spec[term]['gradient_method']['name'])
                grad_method_kwargs = self.loss_spec[term]['gradient_method']['kwargs']
                if self._num_grads_fn is None:
                    if grad_method_kwargs:
                        self._set_attributes(
                            {'_num_grads_fn': partial(numerical_gradients[grad_method_idx], **grad_method_kwargs)}
                        )
                    else:
                        self._set_attributes({'_num_grads_fn': numerical_gradients[grad_method_idx]})
                else:
                    logging.warning(
                        f"Only one gradient computation method can be specified. Raise an issue if you wish to modify "
                        f"this behaviour. Method {self.loss_spec['term']['gradient_method']['name']} was ignored!"
                    )

        # TODO: ALEX: TBD: WE COULD ADD THIS TO THE LOSS SPEC BUT I DON'T SEE THE POINT
        self.blackbox_eval_fcn = watcher_blackbox_wrapper
        # wrap in a decorator that casts the input to np.ndarray and the output to tensor
        wrapper = kwargs.get("blackbox_wrapper")
        self.wrapper = wrapper
        self.predictor = wrapper(self.predictor)

    def _autograd_loss(self):

        # TODO: ADD DOCSTRINGS

        # TODO: CAN YOU RECORD LOSS HERE
        dist_loss = self.distance_fcn(self.instance, self.cf)

        return dist_loss

    def _make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        return self.predictor(X)

    def _get_numerical_gradients(self) -> tf.Tensor:

        # TODO: DOCSTRING (INC. THE ASSUMPTION ON SELF.PREDICTOR TO RETURN SCALAR)
        # shape of `prediction_gradient` is self.cf.shape[0] (batch) x P (n_outputs) x self.cf.shape[1:]
        # (data point shape) 0-index slice below is due to the assumption that `self.predictor` returns a scalar

        blackbox_wrap_fcn_args = (self.predictor, self.target_class, self.instance_class)
        prediction_gradient = self._num_grads_fn(
            self.blackbox_eval_fcn,
            self.copy(self.cf),
            eps=self.eps,
            fcn_args=blackbox_wrap_fcn_args,
        )

        # see docstring to understand the next line
        prediction_gradient = prediction_gradient[:, 0, ...]
        pred = self.blackbox_eval_fcn(self.cf, self.predictor, self.target_class, self.instance_class)
        numerical_gradient = prediction_gradient * self.prediction_grad_fcn(pred, self.target_proba)

        assert numerical_gradient.shape == self.cf.shape

        return numerical_gradient

    def _apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the gradients of the differentiable part of the loss.
        """

        # TODO: THIS METHOD MAKES THE IMPLEMENTATION OF THIS CLASS FRAMEWORK SPECIFIC? IT'S ANNOYING BECAUSE MOST
        #  OF THE CODE IS GOING TO BE EXACTLY THE SAME. CAN WE DO BETTER?

        autograd_grads = gradients[0]
        numerical_grads = self._get_numerical_gradients()
        gradients = [self._mask * (autograd_grads*self.lam + numerical_grads)]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def _record_loss(self):

        # TODO: DOCSTRING
        # cannot call black-box predictor under the gradient tape, so the losses have to
        # be computed outside this context unlike the whitebox case
        model_output = self._make_prediction(self.cf)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        # require to re-evaluate the distance loss to account for the contrib of the numerical gradient
        dist_loss = self.distance_fcn(self.instance, self.cf)
        combined_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self.this_result['distance_loss'] = self.to_numpy_arr(dist_loss).item()
        self.this_result['prediction_loss'] = self.to_numpy_arr(pred_loss).item()
        self.this_result['total_loss'] = self.to_numpy_arr(combined_loss).item()


def _get_search_algorithm(method: str, predictor_type: str, framework: str):

    # TODO: ALEX: DOCSTRING

    if framework == 'tensorflow':
        if method == 'watcher':
            if predictor_type == 'blackbox':
                return WatcherCFSearchBlackBoxTF
            return WatcherCFSearchWhiteBoxTF
        else:
            raise NotImplementedError("Only loss according to Watcher et al is supported at the moment!")
    raise NotImplementedError("PyTorch is not supported at the moment!")


def _get_black_box_wrapper(framework: str) -> Callable:

    # TODO: DOCSTRING

    if framework == 'tensorflow':
        return _wrap_black_box_predictor_tensorflow
    else:
        return _wrap_black_box_predictor_pytorch


def validate_loss_spec(method: str, predictor_type: str, loss_spec: Optional[dict]):

    # TODO: DOCSTRING
    if not loss_spec:
        return
    if method == 'watcher':
        validate_watcher_loss_spec(loss_spec, predictor_type)


def validate_watcher_loss_spec(loss_spec, predictor_type):

    # TODO: ALEX: DOCSTRING

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
            if grad_method_name not in NUMERICAL_GRADIENT_FUNCTIONS:
                raise CounterfactualError(
                    f"Undefined numerical gradients calculation method. Avaialble methods are "
                    f"{NUMERICAL_GRADIENT_FUNCTIONS}"
                )


def infer_device(predictor, framework: str) -> Union[None, str]:

    # TODO: DOCSTRING
    if framework == 'tensorflow':
        return
    default_model_device = next(predictor.parameters()).device
    logging.warning(f"No device specified for the model. Search will take place on {default_model_device}")

    return default_model_device


class Counterfactual(Explainer):

    def __init__(self,
                 predictor,
                 shape,
                 predictor_type: str = 'blackbox',
                 method: str = 'watcher',
                 method_opts: Optional[dict] = None,
                 loss_spec: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework='tensorflow',
                 **kwargs
                 ):

        # TODO: ADD DOCSTRING
        # TODO: REMOVE SHAPE ARGUMENT?
        super().__init__(meta=copy.deepcopy(DEFAULT_META_CF))
        # support for PyTorch models on different devices
        self.framework = framework
        self.predictor = predictor
        model_device = kwargs.get('device', None)
        if not model_device:
            self.model_device = infer_device(predictor, framework)
        else:
            self.model_device = model_device

        self.loss_spec = loss_spec
        validate_loss_spec(method, predictor_type, loss_spec)
        search_algorithm = _get_search_algorithm(method, predictor_type, framework)
        # the black_box_wrapper converts inputs to np.ndarray before calling the predictor
        # and outputs to tensors after predict calls
        black_box_warpper = _get_black_box_wrapper(framework) if predictor_type == 'blackbox' else None
        self._search_algorithm = search_algorithm(
            predictor,
            shape,
            predictor_type=predictor_type,
            loss_spec=loss_spec,
            method_opts=method_opts,
            feature_range=feature_range,
            model_device=self.model_device,
            blackbox_wrapper=black_box_warpper,
        )
        self._search_algorithm.device = self.model_device

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
        """  # noqa W605

        # TODO: A decorator-based soln similar to the numerical gradients can be implemented

        self._check_scale(scale)

        if X is not None:

            if self.scale:
                # infer median absolute deviation (MAD) and update loss
                if scale == 'MAD' or isinstance(scale, bool):
                    scaling_factor = self.mad_scaling(X)
                    self.loss_spec['distance']['kwargs'].update({'feature_scale': scaling_factor})
            if constrain_features:
                # infer feature ranges and update counterfactual constraints
                feat_min, feat_max = np.min(X, axis=0), np.max(X, axis=0)
                self._search_algorithm.cf_constraint = partial(range_constraint, low=feat_min, high=feat_max)

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

    @staticmethod
    def mad_scaling(X: np.ndarray) -> np.ndarray:
        """
        Computes feature-wise mean absolute deviation for `X`. Fixes the latter as an invariant input to the loss
        function, which divides the distance between the current solution and the instance whose counterfactual is
        to be found by this value.
        """

        # TODO: ALEX: TBD: THROW WARNINGS IF THE FEATURE SCALE IS EITHER VERY LARGE OR VERY SMALL?
        # TODO: ALEX: TBD: MOVE TO UTILITY FILE, CREATE A SCALING DECORATOR, AUTOIMPORT THEN APPLY

        feat_median = np.median(X, axis=0)
        mad = np.median(np.abs(X - feat_median), axis=0)

        return mad

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
        if method_opts is not None:
            for key in method_opts:
                if key == 'tol':
                    logger.warning(
                        "tol cannot be overridden at explain time. Please pass method_opts={'tol': 0.5} when "
                        "instantiating the explainer."
                    )
                else:
                    self._search_algorithm._set_attributes(method_opts[key])
        self._search_algorithm._set_attributes(logging_opts)

        if self._search_algorithm.verbose:
            logging.basicConfig(level=logging.DEBUG)

        # find the label and probability of the instance whose CF is searched
        to_tensor = self._search_algorithm.to_tensor
        to_numpy_arr = self._search_algorithm.to_numpy_arr
        y = self._search_algorithm._make_prediction(to_tensor(X))
        instance_class = _convert_to_label(to_numpy_arr(y))
        instance_proba = to_numpy_arr(y[:, instance_class]).item()

        # select features to optimize
        optimized_features = self._select_features(X, feature_whitelist)

        # search for a counterfactual by optimising loss starting from the original solution
        result = self._search_algorithm.counterfactual(
            X,
            instance_class,
            instance_proba,
            optimized_features,
            target_class,
            target_proba=target_proba,
            optimizer=optimizer,
            optimizer_opts=optimizer_opts,
        )

        return self._build_explanation(X, result, instance_class, instance_proba)

    @staticmethod
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
            Set containing the parameters allowed in the update
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

    def _build_explanation(self,
                           X: np.ndarray,
                           result: dict,
                           instance_class: int,
                           instance_proba: float) -> Explanation:
        """
        Creates an explanation object and re-initialises the response to allow calling `explain` multiple times on
        the same explainer.
        """

        # create explanation object
        result['instance'] = X
        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        result['status'] = {'converged': True}
        if not result['cf']:
            result['status']['converged'] = False

        explanation = Explanation(meta=copy.deepcopy(self.meta), data=result)
        # reset response
        self._search_algorithm._initialise_response()

        return explanation

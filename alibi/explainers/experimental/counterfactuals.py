import copy
import logging

import numpy as np
import tensorflow as tf
from inspect import signature

from alibi.api.defaults import DEFAULT_DATA_CF, DEFAULT_META_CF
from alibi.api.interfaces import Explanation, Explainer, FitMixin
from alibi.explainers.backend.common import load_backend
from alibi.explainers.exceptions import CounterfactualError
from alibi.utils.frameworks import infer_device, _check_tf_or_pytorch
from alibi.utils.gradients import numerical_gradients
from alibi.utils.logging import tensorboard_loggers
from alibi.utils.stats import median_abs_deviation
from alibi.utils.wrappers import get_blackbox_wrapper

from collections import defaultdict, namedtuple
from functools import partial
from typing import Any, Callable, Optional, Dict, Set, Tuple, Union, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)

WACTHER_CF_VALID_SCALING = ['median']
WACHTER_CF_PARAMS = ['scale_loss', 'scaling_method' 'constrain_features', 'feature_whitelist', 'fitted']

# define variable tracked for TensorBoard display by specifing tag, data type and description
_CF_WACHTER_TAGS_DEFAULT = [
    'lr',
    'losses/distance_loss',
    'losses/prediction_loss',
    'losses/total_loss',
    'lambda/lambda',
    'lambda/lb',
    'lambda/ub',
    'lambda/delta',
    'counterfactuals/target_class',
    'counterfactuals/target_class_proba',
    'counterfactuals/instance_class_proba',
    'counterfactuals/total_counterfactuals',
    'counterfactuals/current_solution',
]
"""list: This list should be specify the tags under which the variables should be logged. Note that all tags can be 
prefixed at recording time. The code where the variables are created should add the variable name (e.g., delta, lb, ub)
to the data store to write to TensorBoard.
"""
_CF_WACHTER_DATA_TYPES_DEFAULT = ['scalar'] * len(_CF_WACHTER_TAGS_DEFAULT[:-1]) + ['image']
"""list: A list of the data types corresponding to the tags in `_CF_WACHTER_TAGS_DEFAULT`. The following types are 
defined:

    - audio
    - histogram
    - image
    - scalar
    - text
"""
_CF_WACHTER_DESCRIPTIONS_DEFAULT = [
    'Gradient descent optimizer learning rate',
    'Distance between current solution and instance.',
    'Loss between prediction on the current solution and target,',
    '',
    'Current lambda value',
    'Lambda lower bound',
    'Lambda upper bound',
    'Difference between upper and lower bound',
    '',
    '',
    'The probability output by the model for the current solution for the class of the instance whose cf is searched. ',
    '',
    ''
]
"""
list: A list of descriptions corresponding to the tags in `_CF_WACHTER_TAGS_DEFAULT`. A description can be omitted using
''.
"""
_WACHTER_CF_TRACKED_VARIABLES_DEFAULT = {
    'tags': _CF_WACHTER_TAGS_DEFAULT,
    'data_types': _CF_WACHTER_DATA_TYPES_DEFAULT,
    'descriptions': _CF_WACHTER_DESCRIPTIONS_DEFAULT
}
"""
dict: A description of the variables to be recorded to TensorBoard for the WachterCounterfactual class. \
"""
_WACHTER_CF_LOGGING_OPTS_DEFAULT = {
    'verbose': False,
    'log_traces': True,
    'trace_dir': 'logs/cf',
    'summary_freq': 1,
    'image_summary_freq': 10,
    'tracked_variables': _WACHTER_CF_TRACKED_VARIABLES_DEFAULT
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
WACHTER_LAM_OPTS_DEFAULT = {
    'lam_init': 0.1,
    'lams': None,
    'nb_lams': 2,
    'lam_exploration_steps': 20,  # TODO: DOCUMENT THIS PARAM
    'instance_proba_delta': 0.001,  # TODO: DOCUMENT THIS PARAM
    'lam_perc_error': 0.5,  # TODO: DOCUMENT THIS PARAM
    'lam_cf_threshold': 5,
    'lam_multiplier': 10,
    'lam_divider': 10,
    'decay_steps': 10,
    'common_ratio': 10,
    'max_lam_steps': 10,

}
"""dict: The counterfactual search process for the ``'wachter'`` method is guided by the loss 

    .. math:: \ell(X', X, \lambda) = L_{pred} + \lambda L_{dist}

     which depends on the hyperparameter :math:`\lambda`. The explainer will first try to determine a suitable range for 
     :math:`\lambda` and optimize :math:`\lambda` using a bisection algorithm. This dictionary contains the default 
     settings for these methods. These parameters are as follows:

    - `lam_init`: the weight of :math:`L_{dist}`
    - `lams`: counterfactuals exist in restricted regions of the optimisation space, and finding these  regions depends \
    on the :math:`\lambda` paramter. The algorithm first runs an optimisation loop to determine if counterfactuals \
    exist for a given :math:`\lambda`. The default sequence of :math:`\lambda` s is:: 

                        lams = np.array([lam_init / common_ratio ** i for i in range(decay_steps)]) 
    This sequence can be overriden by passing lams directly. For each :math:`\lambda` step, ``max_iter // decay_steps`` \
    iterations of gradient descent updates are performed. The `lams` array should be sorted from high to low.
    - `nb_lams`: Indicates how many values from `lams` need to yield valid counterfactual before the initial search \
    stops. Defaults to 2, which allows finding the tightest optimisation interval for :math:`\lambda`
    - `lam_exploration_steps`: for given values of :math:`\lambda` counterfactuals may not exist. This parameters \
    allows breaking of an optimisation loop early if the probability of the current solution for the maximum probability \
    class predicted on the inititial solution does not change by more than `instance_proba_delta` in \
    `lam_exploration_steps` consecutive iteration. The computational time is reduced as a result.
    - `instance_proba_delta`: the model prediction on the counterfactual for the class with the highest probability for \
    the original instance should decrease by at least this value in `lam_exploration_steps` to warrant continuation of \
    optimisation for a given :math:`\lambda` in the initial search loop
    - `lam_cf_threshold`, `lam_multiplier`, `lam_divider`: see `bisect_lam` docstring in `Watcher counterfactual`
    - `common_ration`, `decay_steps`: role in determining the original optimisation schedule for :math:`\lambda` as  \
    described above
    - 'max_lam_steps': maximum number of times to adjust the regularization constant before terminating the search \
    (number of outer loops).
"""  # noqa W605

WACHTER_SEARCH_OPTS_DEFAULT = {
    'max_iter': 1000,
    'early_stop': 50,
}
"""dict: The default values governing the gradient descent search process for the counterfactual, defined as:

    - 'max_iter': Maximum number of iterations to run the gradient descent for (number of gradient descent loops for \
    each :math:`\lambda` value)
    - 'early_stop': the inner loop will terminate after this number of iterations if either no solutions satisfying \
    the constraint on the prediction are found or if a solution is found at every step for this amount of steps.                
"""  # noqa: W605
WACHTER_METHOD_OPTS = {
    'tol': 0.01,
    'search_opts': WACHTER_SEARCH_OPTS_DEFAULT,
    'lam_opts': WACHTER_LAM_OPTS_DEFAULT,
}
"""dict: Contains the hyperparameters for the counterfactual search. The following keys are defined:

 - `tol`: The algorithm will aim to ensure  :math:`|f_t(X') - p_t| \leq \mathtt{tol}`. Here :math:`f_t(X')` is the \
 :math:`t`th output of the `predictor` on a proposed counterfactual `X'` and `p_t` is a target for said output, \
 specified as `target_proba` when  calling explain.  
- `search_opts`: define termination conditions for the gradient descent process
- `lam_opts`: define the hyperparameters that govern the optimisation for :math:`\lambda`

Any subset of these options can be overridden by passing a dictionary with the corresponding subset of keys to
the explainer constructor or when calling `explain`. If the same subset of arguments is specified in both 
`explain` and the constructor, the `explain` options will override the constructor options. See the documentation for
`WACHTER_SEARCH_OPTS_DEFAULT` and `WACHTER_LAM_OPTS_DEFAULT` to understand algorithm hyperparameters.

Examples
--------
To override the `max_lam_steps` parameters at explain time, call `explain` with the keyword argument::

    method_opts = {'lam_opts':{'max_lam_steps': 50}}.

Similarly, to change the early stopping call `explain` with the keyword argument::

    method_opts = {'search_opts':{'early_stops': 50}}.
"""


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

    # the user does not modify the default loss spec
    if not loss_spec:
        return

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


def _select_features(X: np.ndarray, feature_whitelist: Union[Literal['all'], np.ndarray]) -> np.ndarray:
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


class _WachterCounterfactual:
    """This is a private class that implements the optimization process as described in the `paper`_ by Wachter et al. 
    (2017). The alibi API is implemented in a public class.  

    .. paper: 
       https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf
    """  # noqa

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework: str = 'tensorflow',
                 **kwargs) -> None:
        """
        See public class documentation.
        """

        _check_tf_or_pytorch(framework)
        self.fitted = False
        self.params = {}  # used by public class to update metadata
        model_device = kwargs.get('device', None)
        if not model_device:
            self.model_device = infer_device(predictor, predictor_type, framework)
        else:
            self.model_device = model_device

        optimizer = load_backend(
            explainer_type='counterfactual',
            framework=framework,
            predictor_type=predictor_type,
            method='wachter',
        )
        _validate_wachter_loss_spec(loss_spec, predictor_type)
        blackbox_wrapper = get_blackbox_wrapper(framework) if predictor_type == 'blackbox' else None
        kwargs['blackbox_wrapper'] = blackbox_wrapper
        self.optimizer = optimizer(predictor, loss_spec, feature_range, **kwargs)
        # TODO: DISCUSS HOW THIS SHOULD WORK? CAN THE EXPLAINER SHARE THE GPU WITH THE MODEL?
        self.optimizer.device = self.model_device

        # create attributes and set them with default values
        search_opts = WACHTER_METHOD_OPTS['search_opts']
        lam_opts = WACHTER_METHOD_OPTS['lam_opts']
        logging_opts = _WACHTER_CF_LOGGING_OPTS_DEFAULT
        self._attr_setter(search_opts)
        self._attr_setter(lam_opts)
        self._attr_setter({'tol': WACHTER_METHOD_OPTS['tol']})
        # TODO: ALEX: TBD: ATTRS "POLICE"
        expected_attributes = set(search_opts) | set(lam_opts) | set(logging_opts) | {'tol'}
        expected_attributes |= self.optimizer._expected_attributes
        self.expected_attributes = expected_attributes
        self._set_attributes = partial(self._attr_setter, expected=self.expected_attributes)
        # override defaults with user specification
        if method_opts:
            for key in method_opts:
                if isinstance(method_opts[key], dict):
                    self._set_attributes(method_opts[key])
                else:
                    self._set_attributes({key: method_opts[key]})

        # set default options for logging (can override from wrapper @ explain time)
        self.logging_opts = copy.deepcopy(logging_opts)
        self.log_traces = self.logging_opts['log_traces']
        # logging opts can be overridden so initialisation deferred to explain time
        self.tensorboard = tensorboard_loggers[self.optimizer.framework_name]
        # container for the data logged to tensorboard at every step
        self.data_store = defaultdict(lambda: None)  # type: defaultdict
        self.logger = logger

        # init. at explain time
        self.instance_class = None  # type: Union[int, None]
        self.instance_proba = None  # type: Union[float, None]

        self.step = -1
        self.lam_step = -1
        # return templates
        self.initialise_response()

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)

    def __getattr__(self, key: Any) -> Any:
        return self.__getattribute__(key)

    def _attr_setter(self, attrs: Union[Dict[str, Any], None], expected: Optional[Set[str]] = None) -> None:
        """
        Sets the attributes of the explainer using the (key, value) pairs in attributes. Ignores attributes that
        are not in `expected` if the latter is specified.

        Parameters
        ----------
        attrs
            key-value pairs represent attribute names and values to be set.
        expected
            A dictionary indicating which attributes can be set for the object.
        """

        # TODO: SETATTRIBUTE SHOULD TAKE AN ITERABLE OF KEY-VALUE PAIRS ALSO

        # called with None if the attributes are not overridden
        if not attrs:
            return

        for key, value in attrs.items():
            if expected and key not in expected:
                self.logger.warning(f"Attribute {key} unknown. Attribute will not be set.")
                continue
            self.__setattr__(key, value)
            # sync. setting of variables between base implementation and framework specific functions. Thus, the
            # framework object must explicitly state the attributes that can be overridden at explain time
            if hasattr(self.optimizer, key):
                self.optimizer.__setattr__(key, value)

    def counterfactual(self,
                       instance: np.ndarray,
                       optimised_features: np.ndarray,
                       target_class: Union[Literal['same', 'other'], int] = 'other',
                       target_proba: float = 1.0,
                       optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                       optimizer_opts: Optional[Dict] = None):
        """
        Search for a counterfactual given a starting point (`instance`), the target probability and the target class
        of the counterfactual.

        Parameters
        ----------
        instance
            Search starting point
        optimised_features
            Binary mask. A 1 indicates that the corresponding feature can be changed during search.
        optimizer, optimizer_opts
            See public class `explain` documentation.
        """

        # check inputs
        if instance.shape[0] != 1:
            raise CounterfactualError(
                f"Only single instance explanations supported (leading dim = 1). Got leading dim = {instance.shape[0]}",
            )

        y = self.optimizer.make_prediction(self.optimizer.to_tensor(instance))
        y = self.optimizer.to_numpy_arr(y)
        instance_class = _convert_to_label(y)
        instance_proba = y[:, instance_class].item()
        self.optimizer._get_cf_prediction = partial(self.optimizer.cf_prediction_fcn, src_idx=instance_class)

        self.initialise_variables(
            instance,
            optimised_features,
            target_class,
            instance_class,
            instance_proba,
            target_proba
        )
        self.optimizer.set_optimizer(optimizer, optimizer_opts)
        self._setup_tensorboard()
        result = self.search(init_cf=instance)
        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        self.optimizer.reset_optimizer()
        self._reset_step()

        return result

    def initialise_variables(self,  # type: ignore
                             X: np.ndarray,
                             optimised_features: np.ndarray,
                             target_class: Union[Literal['same', 'other'], int],
                             instance_class: int,
                             instance_proba: float,
                             target_proba: float) -> None:

        """
        Initializes the optimizer variables and sets properties used by the base class for evaluating stopping
        conditions.

        Parameters
        ----------
        X
            Initial condition for the optimization.
        optimised_features, target_proba, target_class
            See`counterfactual` method.
        instance_class, instance_proba
            The class and probability of the prediction for the initial condition.
        """

        self.optimizer.initialise_variables(X, optimised_features, target_class, target_proba, instance_class,
                                            instance_proba)
        self.target_class = target_class
        self.instance = X
        self.instance_class = instance_class
        self.instance_proba = instance_proba

    def search(self, *, init_cf: np.ndarray) -> dict:
        """
        Searches a counterfactual given an initial condition for the counterfactual. The search has two loops:

            - An outer loop, where :math:`\lambda` (the weight of the distance between the current counterfactual and \
            the input `X` to explain in the loss term) is optimised using bisection. See `bisect_lam` method for
            details.

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
        lam_lb, lam, lam_ub = self._initialise_lam(
            self.instance_proba_delta,
            self.nb_lams,
            self.lam_exploration_steps,
            lams=self.lams,
            decay_steps=self.decay_steps,
            common_ratio=self.common_ratio,
        )
        init_lb, init_ub = lam_lb, lam_ub
        self._bisect_lambda = partial(
            self.bisect_lam,
            lam_cf_threshold=self.lam_cf_threshold,
            lam_multiplier=self.lam_multiplier,
            lam_divider=self.lam_divider,
        )

        summary_freq = self.summary_freq
        cf_found = np.zeros((self.max_lam_steps,), dtype=np.uint16)
        # re-init. cf as initial lambda sweep changed the initial condition
        self.optimizer.initialise_cf(init_cf)

        for lam_step in range(self.max_lam_steps):
            self.lam = lam
            self.lam_step += 1
            # re-set learning rate
            self.optimizer.reset_optimizer()
            found, not_found = 0, 0
            for gd_step in range(self.max_iter):
                self.optimizer.cf_step(self.lam)
                self.step += 1
                constraint_satisfied = self.optimizer.check_constraint(
                    self.optimizer.cf,
                    self.optimizer.target_proba,
                    self.tol
                )
                cf_prediction = self.optimizer.make_prediction(self.optimizer.cf)

                # save and optionally display results of current gradient descent step
                current_state = (
                    self.optimizer.to_numpy_arr(self.optimizer.cf),
                    self.optimizer.to_numpy_arr(cf_prediction)
                )
                write_summary = self.log_traces and self.step % summary_freq == 0

                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_response(*current_state)
                    found += 1
                    not_found = 0
                else:
                    found = 0
                    not_found += 1

                if write_summary:
                    self._collect_step_data(*current_state)
                    self.data_store['total_counterfactuals'] = cf_found.sum()
                    self.tensorboard.record(self.step, self.data_store, prefix='counterfactual_search')

                # early stopping criterion - if no solutions or enough solutions found, change lambda
                if found >= self.early_stop or not_found >= self.early_stop:
                    break
            # stop if lam has converged to any of the bounds
            lb_perc_error = 100.0 * np.abs((init_lb - lam) / init_lb)
            ub_perc_error = 100.0 * np.abs((init_ub - lam) / init_ub)
            if np.abs(lb_perc_error - self.lam_perc_error) < 1e-3 or np.abs(ub_perc_error - self.lam_perc_error) < 1e-3:
                break

            lam, lam_lb, lam_ub = self._bisect_lambda(cf_found, lam_step, lam, lam_lb, lam_ub)

        self._display_solution()

        return copy.deepcopy(self.search_results)

    def _initialise_lam(self,
                        instance_proba_delta: float,
                        nb_lams: int,
                        lam_exploration_steps: int,
                        lams: Optional[np.ndarray] = None,
                        decay_steps: int = 10,
                        common_ratio: int = 10):  # type: ignore
        """
        Runs a search procedure over a specified range of lambdas in order to determine a range of values of this
        parameter for which counterfactuals exist. If `lams` is not specified, a range of lambda is created by diving an
        initial value by the geometric sequence ``[decay_factor**i for i in range(decay_steps)]``. The method saves the
        results of this sweep in the `'lambda_sweep'` field of the explanation 'data' field.

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
        A tuple containing:

            - 'lb': lower bound for lambda
            - 'ub': upper bound for lambda
            - 'midpoint': the midpoints of the interval [lb, ub]
        """

        n_steps = self.max_iter // decay_steps
        if lams is None:
            lams = np.array([self.lam_init / common_ratio ** i for i in range(decay_steps)])  # exponential decay
        lams[::-1].sort()
        cf_found = np.zeros(lams.shape, dtype=np.uint16)

        for lam_step, lam in enumerate(lams):
            explored_steps = 0
            # optimiser is re-created so that lr schedule is reset for every lam
            self.optimizer.reset_optimizer()
            self.lam = lam
            self.data_store['lambda'] = lam
            self.lam_step += 1
            for gd_step in range(n_steps):
                # update cf with loss gradient for a fixed lambda
                self.optimizer.cf_step(self.lam)
                self.step += 1
                # NB: a bit of a weird pattern but kept it like this for clarity
                constraint_satisfied = self.optimizer.check_constraint(
                    self.optimizer.cf,
                    self.optimizer.target_proba,
                    self.tol
                )
                cf_prediction = self.optimizer.make_prediction(self.optimizer.cf)
                instance_class_pred = self.optimizer.to_numpy_arr(cf_prediction[:, self.instance_class]).item()

                # update response and log data to TensorBoard
                write_summary = self.log_traces and self.step % self.summary_freq == 0
                current_state = (
                    self.optimizer.to_numpy_arr(self.optimizer.cf),
                    self.optimizer.to_numpy_arr(cf_prediction)
                )

                if write_summary:
                    self._collect_step_data(*current_state)
                    total_counterfactuals = cf_found.sum() + 1 if constraint_satisfied else cf_found.sum()
                    self.data_store['total_counterfactuals'] = total_counterfactuals
                    self.tensorboard.record(self.step, self.data_store, prefix='lambda_sweep')

                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_response(*current_state)
                    # sufficient to find a counterfactual for a given lambda in order to consider that lambda valid
                    break

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
        # determine upper and lower bounds given the solutions found
        lam_bounds = self.compute_lam_bounds(cf_found, lams)

        # re-initialise response s.t. 'cf' and 'all' fields contain only main search results
        sweep_results = copy.deepcopy(self.search_results)
        self.initialise_response()
        self.search_results['lambda_sweep']['all'] = sweep_results['all']
        self.search_results['lambda_sweep']['cf'] = sweep_results['cf']
        self._reset_step()
        self.data_store['lb'] = lam_bounds.lb
        self.data_store['lambda'] = lam_bounds.midpoint
        self.data_store['ub'] = lam_bounds.ub

        return lam_bounds

    def compute_lam_bounds(self, cf_found: np.ndarray, lams: np.ndarray):  # type: ignore
        """
        Determine an upper and lower bound for :math:`\lambda`.

        Parameters
        ----------
        cf_found
            A binary 1D-array, where each position corresponds to a :math:`\lambda` value. A `1` indicates that a
            counterfactual was found for that values of :math:`\lambda`
        lams
            An array of the same shape as `cf_found`, containing the :math:`\lambda` values for which counterfactuals
            were searched.

        Returns
        -------
        bounds
            A a named tuple with fields:

                - 'lb': lower bound for lambda
                - 'ub': upper bound for lambda
                - 'midpoint': the midpoint of the interval :math:`[lb, ub]`

        """  # noqa W605

        self.logger.debug(f"Counterfactuals found: {cf_found}")
        lam_bounds = namedtuple('lam_bounds', 'lb midpoint ub')
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
            bounds = lam_bounds(lb=lams[0], midpoint=0.5 * (lams[0] + lams[1]), ub=lams[1])

        elif cf_found.sum() == 1:
            # this case is unlikely to occur in practice
            if cf_found[0] == 1:
                bounds = lam_bounds(lb=lams[0], midpoint=5.5 * lams[0], ub=10 * lams[0])
            else:
                valid_lam_idx = np.nonzero(cf_found)[0][0]
                lb = lams[valid_lam_idx]
                ub = lams[valid_lam_idx - 1]
                bounds = lam_bounds(lb=lb, midpoint=0.5 * (lb + ub), ub=ub)
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
            lb = lams[lam_lb_idx]
            ub = lams[lam_ub_idx]
            bounds = lam_bounds(lb=lb, midpoint=0.5 * (lb + ub), ub=ub)

        self.logger.debug(f"Found upper and lower bounds for lambda: {bounds.ub}, {bounds.lb}")

        return bounds

    def initialise_response(self) -> None:
        """
        Initialises the templates that will form the body of the `explanation.data` field.
        """

        self.step_data = [
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

        self.search_results = copy.deepcopy(DEFAULT_DATA_CF)
        self.search_results['all'] = defaultdict(list)
        self.search_results['lambda_sweep'] = {}

    def bisect_lam(self,
                   cf_found: np.ndarray,
                   lam_step: int,
                   lam: float,
                   lam_lb: float,
                   lam_ub: float,
                   lam_cf_threshold: int = 5,
                   lam_multiplier: int = 10,
                   lam_divider: int = 10) -> Tuple[float, float, float]:
        """
        Runs a bisection algorithm to optimise :math:`lambda`, which is adjust according to the following algorithm:

            - if the number of counterfactuals exceeds `lam_cf_threshold`, then:
                * the :math:`\lambda` lower bound, :math:`\lambda_{lb}`, is set to :math:`\max(\lambda, \lambda_{lb})`
                * if :math:`\lambda < 10^9` then :math:`\lambda \gets 0.5*(\lambda_{lb}, \lambda_{ub})`
                * else :math:`\lambda` is mutiplied by `lambda_multiplier`
            - if the number of counterfactuals is below `lam_cf_threshold`, then:
                * the :math:`\lambda` upper bound, :math:`\lambda_{ub}`, is set to :math:`\min(\lambda, \lambda_{ub})`
                * if :math:`\lambda > 0` then :math:`\lambda \gets 0.5*(\lambda_{lb}, \lambda_{ub})`
                * else :math:`\lambda` is divided by `lambda_divider`

            See `WACHTER_METHOD_OPTS` for details parameters defaults.

        Parameters
        ----------
        cf_found
            Array containing the number of counterfactuals found for a given value of :math:`\lambda` in each position.
        lam_step
            The current optimisation step for :math:`\lambda`.
        lam_lb, lam_ub
            The lower and upper bounds for the :math:`\lambda`.
        lam_cf_threshold, lam_divider, lam_multiplier
            Parameters that control the bisection algorithm as described above.

        Returns
        -------
        lam, lam_lb, lam_ub
            Updated values for :math:`\lambda` and its lower and upper bound.
        """  # noqa W605

        # lam_cf_threshold: minimum number of CF instances to warrant increasing lambda
        if cf_found[lam_step] >= lam_cf_threshold:
            lam_lb = max(lam, lam_lb)
            self.logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_ub < 1e9:
                lam = (lam_lb + lam_ub) / 2
            else:
                lam *= lam_multiplier
                self.logger.debug(f"Changed lambda to {lam}")

        elif cf_found[lam_step] < lam_cf_threshold:
            # if not enough solutions found so far, decrease lambda by a factor of 10,
            # otherwise bisect up to the last known successful lambda
            lam_ub = min(lam_ub, lam)
            self.logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_lb > 0:
                lam = (lam_lb + lam_ub) / 2
                self.logger.debug(f"Changed lambda to {lam}")
            else:
                lam /= lam_divider

        self.data_store['lambda'] = lam
        self.data_store['lb'] = lam_lb
        self.data_store['ub'] = lam_ub
        self.data_store['delta'] = lam_ub - lam_lb

        return lam, lam_lb, lam_ub

    def _collect_step_data(self, current_cf: np.ndarray, current_cf_pred: np.ndarray) -> None:
        """
        Updates the data store with information from the current optimisation step.

        Parameters
        ----------
        current_cf, current_cf_pred
            See ``_update_response`` method.
        """

        # collect optimizer state from the framework
        # important that the optimizer state dict keys match the variable names the logger knows about
        opt_state = self.optimizer.collect_step_data()
        self.data_store.update(opt_state)
        # compute other state from information available to this function
        pred_class = _convert_to_label(current_cf_pred)
        target_pred_proba = current_cf_pred[:, pred_class].item()
        instance_class_proba = current_cf_pred[:, self.instance_class].item()
        self.data_store['target_class'] = pred_class
        self.data_store['target_class_proba'] = target_pred_proba
        self.data_store['instance_class_proba'] = instance_class_proba
        self.data_store['current_solution'] = current_cf

    def _update_response(self, current_cf: np.ndarray, current_cf_pred: np.ndarray) -> None:
        """
        Updates the model response. Called only if current solution, :math:`X'` satisfies 
        :math:`|f_t(X') - p_t| < \mathtt{tol}`. Here :math:`f_t` is the model output and `p_t` is the target model 
        output.

        Parameters
        ----------
        current_cf
            The current solution, :math:`X'`.
        current_cf_pred
            The model prediction on `current_cf`.

        """  # noqa W605

        # collect data from the optimizer
        opt_data = self.optimizer._get_current_state()
        # augment the data and update the response
        pred_class = _convert_to_label(current_cf_pred)
        target_pred_proba = current_cf_pred[:, pred_class].item()
        instance_class_proba = current_cf_pred[:, self.instance_class].item()

        this_result = dict.fromkeys(self.step_data)
        this_result.update(
            {
                'distance_loss': opt_data['distance_loss'],
                'prediction_loss': opt_data['prediction_loss'],
                'total_loss': opt_data['total_loss'],
                'lambda': self.lam,
                'target_class': pred_class,
                'target_class_proba': target_pred_proba,
                'instance_class_proba': instance_class_proba,
                'X': current_cf,
                'step': self.step,
            }
        )
        self.search_results['all'][self.lam_step].append(this_result)

        # update best CF if it has a smaller distance
        if not self.search_results['cf']:
            self.search_results['cf'] = this_result
        elif this_result['distance_loss'] < self.search_results['cf']['distance_loss']:
            self.search_results['cf'] = this_result

        self.logger.debug(f"CF found at step {self.step}.")

    def _reset_step(self):
        """
        Resets the optimisation step for gradient descent and for the weight optimisation step (`lam_step`).
        """
        self.step = -1
        self.lam_step = -1

    def _setup_tensorboard(self):
        """
        Initialises the TensorBoard writer.
        """

        self._set_attributes(self.logging_opts)
        if self.log_traces:
            self.tensorboard = self.tensorboard().setup(self.logging_opts)

    def _display_solution(self) -> None:
        """
        Displays the instance along with the counterfactual with the smallest distance from the instance. Used for image
        data only.
        """

        # only for images and when logging to TensorbBoard is active
        if not self.log_traces or len(self.instance.shape) != 4:
            return

        soln_description = r"A counterfactual `X'` that satisfies `|f(X') - y'| < \epsilon` and minimizes `d(X, X')`." \
                           r" Here `y'` is the target probability and `f(X')` is the model prediction on the " \
                           r"counterfactual. Found at optimisation step {}.".format(
            self.search_results['cf']['step'])  # noqa
        self.tensorboard.record_step(
            step=0,
            tag='optimal_solution/cf',
            value=self.search_results['cf']['X'],
            data_type='image',
            description=soln_description,
        )

        self.tensorboard.record_step(
            step=0,
            tag='optimal_solution/instance',
            value=self.instance,
            data_type='image',
            description='Original instance'
        )

    def fit(self,
            X: Optional[np.ndarray] = None,
            scale: Union[Literal['median'], bool] = False,
            constrain_features: bool = True) -> "_WachterCounterfactual":
        """
        See public class documentation
        """

        # TODO: A decorator-based soln similar to the numerical gradients can be implemented for scaling

        self._check_scale(scale)

        if X is not None:

            if self.scale:
                # infer median absolute deviation (MAD) and update loss
                if scale == 'median' or isinstance(scale, bool):
                    scaling_factor = median_abs_deviation(X)
                    self.optimizer.distance_fcn.keywords['feature_scale'] = scaling_factor

            if constrain_features:
                # infer feature ranges and update counterfactual constraints
                feat_min, feat_max = np.min(X, axis=0), np.max(X, axis=0)
                self.optimizer.cf_constraint = [feat_min, feat_max]

        self.fitted = True
        scaling_method = 'median' if self.scale else 'N/A'
        self.params = {
            'scale_loss': self.scale,
            'scaling_method': scaling_method,
            'constrain_features': constrain_features,
            'fitted': self.fitted,
        }

        return self

    def _check_scale(self, scale: Union[Literal['median'], bool]) -> None:
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
            loss_params = signature(self.optimizer.loss_spec['distance']).parameters
            if 'feature_scale' not in loss_params:
                logger.warning(
                    f"Scaling option specified but the loss specified did not have a parameter named 'feature_scale'. "
                    f"Scaling will not be applied!"
                )
                scale_ = False

        self.scale = scale_  # type: bool


class WachterCounterfactual(Explainer, FitMixin):

    # TODO: ALEX: TBD: AS WE IMPLEMENT COUNTERFACTUALS, WE MAY REALISE THIS IS COMMON AND WOULD DO THE FOLLOWING:
    #  - MAKE THIS A SUPERCLASS (e.g., CounterfactualAPI). The public classes (e.g., WachterCounterfactual) inherit from
    #  it to show the docs specific to the method and potentially override some behaviour (eg. fit might have other
    #  set of arguments, etc)
    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework: str = 'tensorflow',
                 **kwargs) -> None:
        """
        Counterfactual explanation method based on `Wachter et al. (2017)`_ (pp. 854). 

        .. _Wachter et al. (2017): 
           https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf

        Parameters
        ----------
        predictor
            A model which can be implemented in TensorFlow, PyTorch, or a callable that can be queried with a 
            `np.ndarray` of instances in order to return predictions. The object returned (which is a framework-specific 
            tensor) should always be two-dimensional. For example, a single-output classification model operating on a 
            single input instance should return a tensor or array with shape  `(1, 1)`. The explainer assumes the 
            `predictor` returns probabilities. In the future this explainer may be extended to work with regression 
            models.
         predictor_type: {'blackbox', 'whitebox'}

            - 'blackbox' indicates that the algorithm does not have access to the model parameters (e.g., the predictor \
            is an API endpoint so can only be queried to return prediction). This argument should be used to search for \
            counterfactuals of non-differentiable models (e.g., trees, random forests) or models implemented in machine \
            learning frameworks other than PyTorch and TensorFlow.

            - 'whitebox' indicates that the model is implemented in PyTorch or TensorFlow and that the explainer has 
            access to the parameters, so that the automatic differentiation and optimization of the framework can be 
            leveraged in the search process.  

        loss_spec
            Check the `pytorch`_ or `tensorflow`_ loss specification for detailed explanation of the assumptions of the 
            loss function. As detailed there, the loss_specification depends on the `predictor_type`. This argument 
            should be used by advanced users who want to modify the functional form of the loss term  - the positional
            arguments for the individual terms *_should be the same as in the default specification_*. 

            .. _pytorch: file:///C:/Users/alexc/dev/alibi/doc/_build/html/api/alibi.explainers.experimental.counterfactuals.html
            .. _tensorflow: file:///C:/Users/alexc/dev/alibi/doc/_build/html/api/alibi.explainers.backend.tensorflow.counterfactuals.html

        method_opts
            Paramaters that control the optimisation process. See documentation for `WATCHER_METHOD_OPTS` above for 
            detailed information about the role of each parameter and how to override them through the `explain` 
            interface.. It is recommended that the user runs the algorithm with the defaults and then and uses the 
            TensorBoard display to adjust  the parameters (by overriding them through the explain interface) in cases of 
            non-convergence, long-running explanations or if different explanation properties are desired. 
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
            ``'tensorflow'`` is a valid version for the current release.

        kwargs
            Valid kwargs include:
                - `model_device`: used to pass a model device so that cpu/gpu computation can be supported for PyTorch     
        """  # noqa W605

        # TODO: UPDATE LOSS SPEC HYPERLINKS TO POINT TO THE READTHEDOCS PART OF THE ADDRESS
        super().__init__(meta=copy.deepcopy(DEFAULT_META_CF))

        self._explainer_type = _WachterCounterfactual
        explainer_args = (predictor,)
        explainer_kwargs = {
            'predictor_type': predictor_type,
            'loss_spec': loss_spec,
            'method_opts': method_opts,
            'feature_range': feature_range,
            'framework': framework,
        }
        self._explainer = self._explainer_type(*explainer_args, **explainer_kwargs, **kwargs)

    def fit(self,
            X: Optional[np.ndarray] = None,
            scale: Union[Literal['median'], bool] = False,
            constrain_features: bool = True) -> "WachterCounterfactual":
        """
        Calling this method with an array of :math:`N` data points, assumed to be the leading dimension of `X`, has the
        following functions:

            - If `constrain_features=True`, the minimum and maximum of the array along the leading dimension constrain \
            the minimum and the maximum of the counterfactual
            - If the `scale` argument is set to `True` or `median`, then the distance between the input and the \
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

        # TODO: ALEX: TBD: Should fit be part of the private class? We could also defer the call.
        # TODO: A decorator-based soln similar to the numerical gradients can be implemented for scaling
        self._explainer.fit(X=X, scale=scale, constrain_features=constrain_features)
        self._update_metadata(self._explainer.params, params=True, allowed=set(WACHTER_CF_PARAMS))

        return self

    def explain(self,
                X: np.ndarray,
                target_class: Union[Literal['same', 'other'], int] = 'other',
                target_proba: float = 1.0,
                optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                optimizer_opts: Optional[Dict] = None,
                feature_whitelist: Union[Literal['all'], np.ndarray] = 'all',
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
            ``, with the following structure::

                {
                    'verbose': False,
                    'log_traces': True,
                    'trace_dir': 'logs/cf',
                    'summary_freq': 1,
                    'image_summary_freq': 10,
                    'tracked_variables': {'tags': [], 'data_types': [], 'descriptions': []},
                }

                Default values for `verbose` and `log_traces` are as shown above.

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

        # TODO: UPDATE DOCS

        # override default method settings with user input
        if method_opts:
            for key in method_opts:
                if isinstance(method_opts[key], Dict):
                    self._explainer._set_attributes(method_opts[key])
                else:
                    self._explainer._set_attributes({key: method_opts[key]})

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
        specified in the `allowed` dictionary

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

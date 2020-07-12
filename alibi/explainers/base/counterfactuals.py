import copy
import logging

import numpy as np
import tensorflow as tf

from alibi.api.defaults import DEFAULT_DATA_CF
from alibi.explainers.backend.common import load_backend
from alibi.explainers.exceptions import CounterfactualError
from alibi.utils.logging import tensorboard_loggers
from alibi.explainers.base import register_explainer
from collections import defaultdict, namedtuple
from functools import partial
from typing import Union, Callable, Optional, Dict, Tuple, Any, Set


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
""" # noqa W605

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
`explain` and the constructor, the `explain` options will override the constructor options.

Examples
--------
To override the `max_lam_steps` parameters at explain time, call `explain` with the keyword argument::

    method_opts = {'lam_opts':{'max_lam_steps': 50}}.

Similarly, to change the early stopping call `explain` with the keyword argument::
    
    method_opts = {'search_opts':{'early_stops': 50}}.
"""


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


@register_explainer(explainer_type='counterfactual', method='wachter')
class WachterCounterfactual:

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 predictor_type: str = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework: str = 'tensorflow',
                 logger: logging.Logger = None,
                 **kwargs) -> None:
        """
        Counterfactual explanation method based on `Wachter et al. (2017)`_ (pp. 854). The role and usage of arguments 
        not specified is detailed in the `wrapper`_ class.
        
        .. _wrapper: file:///C:/Users/alexc/dev/alibi/doc/_build/html/api/alibi.explainers.experimental.counterfactuals.html

        .. _Wachter et al. (2017): 
           https://jolt.law.harvard.edu/assets/articlePDFs/v31/Counterfactual-Explanations-without-Opening-the-Black-Box-Sandra-Wachter-et-al.pdf

        Parameters
        ----------
        method_opts
            Paramaters that control the optimisation process. See documentation for `WATCHER_METHOD_OPTS` above for 
            detailed information about the role of each parameter. 
        loss_spec
            Check the `pytorch`_ or `tensorflow`_ loss specification for detailed explanation of the assumptions of the 
            loss function.
            
            .. _pytorch: file:///C:/Users/alexc/dev/alibi/doc/_build/html/api/alibi.explainers.experimental.counterfactuals.html
            .. _tensorflow: file:///C:/Users/alexc/dev/alibi/doc/_build/html/api/alibi.explainers.backend.tensorflow.counterfactuals.html
        logger
            A logger passed by the wrapper class in order to collect debugging messages.
        kwargs
            Valid kwargs include:
                - `model_device`: used to pass a model device so that cpu/gpu computation can be supported for PyTorch
                - blackbox_wrapper: a decorator that the wrapper passes to the algorithm so that the I/O of the \
                black-box model can be converted to `np.ndarray` or framework specific tensors.         
        """  # noqa W605

        self.fitted = False

        optimizer = load_backend(
            explainer_type='counterfactual',
            framework=framework,
            predictor_type=predictor_type,
            method='wachter',
        )
        self.optimizer = optimizer(predictor, loss_spec, feature_range, **kwargs)

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
        are not specified in CF_ATTRIBUTES.

        Parameters
        ----------
        attrs
            key-value pairs represent attribute names and values to be set.
        expected
            A
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
                       target_class: Union[str, int] = 'other',
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
            See wrapper documentation
        """

        # TODO: ADD HYPERLINK TO WRAPPER

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
        result = self.optimize(init_cf=instance)
        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        self.optimizer.reset_optimizer()
        self._reset_step()

        return result

    def initialise_variables(self,   # type: ignore
                             X: np.ndarray,
                             optimised_features: np.ndarray,
                             target_class: Union[int, str],
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

    def optimize(self, *, init_cf: np.ndarray) -> dict:
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
        cf_found = np.zeros((self.max_lam_steps, ), dtype=np.uint16)
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
            lb_perc_error = 100.0*np.abs((init_lb - lam)/init_lb)
            ub_perc_error = 100.0*np.abs((init_ub - lam)/init_ub)
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
            bounds = lam_bounds(lb=lams[0], midpoint=0.5*(lams[0] + lams[1]),  ub=lams[1])

        elif cf_found.sum() == 1:
            # this case is unlikely to occur in practice
            if cf_found[0] == 1:
                bounds = lam_bounds(lb=lams[0], midpoint=5.5*lams[0], ub=10 * lams[0])
            else:
                valid_lam_idx = np.nonzero(cf_found)[0][0]
                lb = lams[valid_lam_idx]
                ub = lams[valid_lam_idx - 1]
                bounds = lam_bounds(lb=lb, midpoint=0.5*(lb + ub), ub=ub)
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
            bounds = lam_bounds(lb=lb, midpoint=0.5*(lb + ub), ub=ub)

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
                           r"counterfactual. Found at optimisation step {}.".format(self.search_results['cf']['step']) # noqa
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

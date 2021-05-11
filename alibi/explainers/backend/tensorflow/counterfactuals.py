import numpy as np
import tensorflow as tf

from alibi.explainers.backend import register_backend
from alibi.explainers.exceptions import CounterfactualError
from alibi.utils.gradients import get_numerical_gradient
from alibi.utils.wrappers import get_blackbox_wrapper
from collections import defaultdict
from copy import deepcopy
from functools import partial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, TYPE_CHECKING
from typing_extensions import Final, Literal

if TYPE_CHECKING:
    import keras


def scaled_l1_loss(instance: tf.Tensor, cf: tf.Variable, feature_scale: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Scaled :math:`\ell_{1}` loss between an instance and a counterfactual. The distance is computed feature-wise and 
    a sum reduction is applied to the result.
    
    Parameters
    ----------
    instance
        Instance. 
    cf
        Counterfactual for instance.
    feature_scale
        If passed, the feature-wise distance will be divided by a scaling factor prior to summation.
    """  # noqa W605

    ax_sum = tuple(np.arange(1, len(instance.shape)))
    if feature_scale is None:
        return tf.reduce_sum(tf.abs(cf - instance), axis=ax_sum, name='l1')
    return tf.reduce_sum(tf.abs(cf - instance) / feature_scale, axis=ax_sum, name='l1')


def squared_loss(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:
    """
    Squared loss function.

    Paramters
    --------
    pred_probas
        Predicted probabilities.
    target_probas
        Target probabilities.
    """
    return tf.square(pred_probas - target_probas)


def squared_loss_grad(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:
    """
    Gradient of the squared loss.

    Parameters
    pred_probas, target_probas
        See `squared_loss` documentation.
    """
    return 2 * (pred_probas - target_probas)


def wachter_loss(distance: tf.Tensor, lam: float, pred: tf.Tensor) -> tf.Tensor:
    """
    A function that combines the weighted distance loss with prediction loss.

    Paramaters
    ----------
    distance
        Distance loss.
    lam
        Distance loss weight.
    pred
        Prediction loss.
    """
    return lam * distance + pred


WACHTER_LOSS_SPEC_WHITEBOX = {
    'prediction': {'fcn': squared_loss},
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'loss': {'fcn': wachter_loss},  # function that combines the prediction and distance
}  # type: Dict[str, Mapping[str, Any]]
"""
dict: A specification that allows customising the Wachter loss, defined as:

.. math:: L_{pred} + \lambda L_{dist}.

See our documentation for `details`_.

.. _details: https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html

This specification assumes that the search algorithm has access to the parameters of the predictor.

To specify :math:`L_{pred}`, the ``'prediction'`` field of the specification should have a callable under the ``'fcn'``
and its kwargs under the ``'kwargs'``. The callable arguments are assumed to be the model prediction for the specified
class and the target for the said prediction, specified as an input to `explain`, respectively.

Similarly, :math:`L_{dist}` is specified in the ``'distance'`` field of the specification. The args of the callable are
assumed to be :math:`X` the instance whose counterfactual is searched and the counterfactual at the current iteration
:math:`X'`, respectively.

The ``'loss'`` field should be a function that combines the outputs of the prediction and distance parts (``'fcn'``
field) along with its kwargs.

"""  # noqa: W605
WACHTER_LOSS_SPEC_BLACKBOX = {
    'prediction': {
        'fcn': squared_loss,
        'pred_out_grad_fcn': squared_loss_grad,
        'pred_out_grad_fcn_kwargs': {},
    },
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'loss': {'fcn': wachter_loss},  # function that combines the prediction and distance
    'num_grad_method': {'name': 'central_difference', 'kwargs': {'eps': 0.01}}
}  # type: Dict[str, Mapping[str, Any]]
"""dict: A specification that allows customising the Wachter loss, defined as:

.. math:: L_{pred} + \lambda L_{dist}

See our `documentation`_ for details.

.. _documentation: https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html


This specification assumes that the term specified in the ``'prediction'`` field cannot be differentiated using
automatic differentiation frameworks. Consequently, this term requires additional information in the following fields:

    - ``'num_grad_method'``. The ``'name'`` sub-field specifies a function that can compute numerical gradients given \
    a black-box function and an input tensor :math:`X`. See `alibi.utils.tensorflow.gradients` for an example and \
    interface description and `alibi.explainers.backend.tensorflow.counterfactuals` `get_numerical_gradients` method \
    for an integration example. The kwargs of this function should be specified in the ``'kwargs'`` field

    - ``'pred_out_grad_fcn'`` should be a callable taking the same inputs as ``'fcn'``. Represents the gradient of the \
    prediction term wrt model output. The output of this function is multiplied by the numerical gradient to give \
    the gradient of the non-diferentiable wrt to the input

    - ``'pred_out_grad_fcn_kwargs'`` specify the kwargs of ``'grad_fcn'``.
"""  # noqa: W605


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
    ValueError
        If `low` > `high`.
    """

    feat_interval = high - low
    if np.any(feat_interval < 0.):
        raise ValueError(
            f"Invalid constrain. Lower bound needs to be smaller upper bound but the difference was "
            f"{feat_interval} for an element.",
        )

    return tf.clip_by_value(X, clip_value_min=low, clip_value_max=high)


def slice_prediction(prediction: tf.Tensor,
                     target_idx: Union[int, Literal['same', 'other']],
                     src_idx: int) -> tf.Tensor:
    """
    Returns a slice from `prediction` depending on the value of the `target_idx`.

    Parameters
    ----------
    prediction
        A prediction tensor
    target_idx

        - If ``int``, the slice of `prediction` indicated by it is returned
        - If 'same', the slice indicated by `src_idx` is returned
        - If 'other', the slice that maximises the prediction other than `src_idx` is returned

    src_idx
        An index in the prediction that influences the returned slice as described above.
    """

    if isinstance(target_idx, int) or target_idx == 'same':
        if isinstance(target_idx, str):
            return prediction[:, src_idx]
        return prediction[:, target_idx]

    if target_idx != 'other':
        raise ValueError("The only allowed values for output slices are {'same','other'}")

    _, indices = tf.math.top_k(prediction, k=2)
    if indices[0][0] == src_idx:
        cf_prediction = prediction[:, indices[0][1].numpy()]
    else:
        cf_prediction = prediction[:, indices[0][0].numpy()]
    return cf_prediction


def wachter_blackbox_wrapper(X: Union[tf.Tensor, tf.Variable],
                             predictor: Callable,
                             target_idx: Union[int, Literal['same', 'other']],
                             src_idx: int) -> tf.Tensor:
    """
    A wrapper that slices a vector-valued predictor, turning it to a scalar-valued predictor. The scalar depends on the
    `target_idx` and `src_idx`. The predictor (and the input) should be two-dimensional. Therefore, if a record of
    dimension :math:`D` is input, it should be passed as a tensor of shape `(1, D)`. A single-output classification
    model operating on this  instance should return a tensor or array with shape  `(1, 1)`.

    Parameters
    ----------
    X
        Predictor input
    predictor
        Predictor to be wrapped.
    target_idx, src_idx
        See _slice_prediction documentation.
    """

    prediction = predictor(X)
    pred_target = slice_prediction(prediction, target_idx, src_idx)
    return pred_target


class TFGradientOptimizer:
    """A base class from which all optimizers used to search for counterfactuals should inherit. It implements basic
    functionality such as:

        - setting the loss terms as optimizer attributes
        - TensorFlow optimizer setter
        - `step`, a method that differentiates the loss function specified and updates the solution
        - methods that copy or convert to/from tf.Tensor objects, that can be used by calling objects to implement \
        framework-independent functionality
        - a method that updates and returns the state dictionary to the calling object (used for logging purposes)
    """
    framework = 'tensorflow'
    num_grad_method = 'central_difference'
    num_grad_method_kwargs = {'eps': 0.01}

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model],
                 loss_spec: Dict[str, Dict[str, Any]],
                 predictor_type: Literal['blackbox', 'whitebox'],
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 ) -> None:
        """
        Initializes the optimizer by:

            - Setting the callables in the `loss_spec` as object attributes, so that they can be accessed in methods \
            which compute the loss function
            - Setting placeholders for quantities that all algorithms need (e.g., counterfactual, target probability, \
            instance and its class, optimizer and its copy, counterfactual constraints, device)

        Parameters
        ----------
        predictor
            A predictor which will be queried in order to find a counterfactual for a given instance. See documentation
            for specific implementations (e.g., alibi.explainers.experimental.counterfactuals.WatcherCounterfactual) for
            more details about the predictor requirements.
        loss_spec
            A dictionary with the structure::

                {
                'loss_term_name_1': {'fcn': Callable, 'kwargs': {'eps': 0.01}}
                'loss_term_name_2': {'fcn': Callable, 'kwargs': {}}
                'loss':  {'fcn': Callable, 'kwargs': {'alpha': 0.1, 'beta': 0.5}
                }

            Each key in the dictionary is the name of a term of the loss to be implemented (e.g., `loss_term_name_1` in
            the example above). The mapping of each term is expected to  have a ``'fcn'`` key where a callable
            implementing the loss term is stored and a ``'kwargs'`` entry where keyword arguments for the said callable
            are passed. If no kwargs are needed, then the latter should be set to {}. Partial functions obtained from the
            callables (or the callables themselves when there are no kwargs  specified) as object attributes under
            the name `*_fcn` where the * is substituted by the name of the term. For the example above, the object will
            have attributes `loss_term_name_1`, `loss_term_name_2` and `loss`.

            The object will also contain a `state` property, a mapping where the values of the loss terms can be stored
            as they are computed in order to be logged to TensorBoard automatically.
        predictor_type
            `blackbox` or `whitebox`, used for configuring how and when numerical gradients are computed.
        feature_range
            Used to constrain the range of the counterfactual features. It should be specified such that it can be
            compared element-wise with a counterfactual. See documentation for specific counterfactual implementations
            (e.g., alibi.explainers.experimental.counterfactuals.WatcherCounterfactual) for more details about feature
            range constraints.
        """
        self.predictor_type = predictor_type
        self._set_predictor(predictor)

        if self.predictor_type == 'blackbox':
            # set numerical gradient scheme
            try:
                num_grad_method = loss_spec['num_grad_method']['name']
                kwargs = loss_spec['num_grad_method'].get('kwargs', {})
                loss_spec.pop('num_grad_method')
            except KeyError:
                num_grad_method = self.num_grad_method
                kwargs = self.num_grad_method_kwargs

            try:
                num_grad_fcn = get_numerical_gradient(self.framework, num_grad_method)
            except KeyError:
                raise CounterfactualError(f'Unknown numerical differention scheme: {num_grad_method}')

            setattr(self, 'num_grad_fcn', partial(num_grad_fcn, **kwargs))

        # set losses as attributes
        for term in loss_spec:
            kwargs = loss_spec[term].get('kwargs', {})
            fcn = partial(loss_spec[term]['fcn'], **kwargs)
            setattr(self, f'{term}_fcn', fcn)

            # set fcn to calculate the prediction term gradient wrt to predictor output
            if 'pred_out_grad_fcn' in loss_spec[term]:
                grad_kwargs = loss_spec[term]['pred_out_grad_fcn_kwargs'].get('kwargs', {})
                grad_fcn = partial(loss_spec[term]['pred_out_grad_fcn'], **grad_kwargs)
                setattr(self, f'{term}_grad_fcn', grad_fcn)

        # the algorithm state is by default each loss term specified
        # add scalars to be logged to TensorBoard to this dictionary
        self.state = defaultdict(lambda: None, dict.fromkeys(loss_spec.keys(), 0.0))

        # updated at explain time
        self.optimizer = None  # type: Union[tf.keras.optimizers.Optimizer, None]
        self._optimizer_copy = None

        # problem variables: init. at explain time
        self.target_proba = None  # type: Union[tf.Tensor, None]
        self.instance = None  # type: Union[tf.Tensor, None]
        self.instance_class = None  # type: Union[int, None]
        self.solution = None  # type: Union[tf.Variable, None]
        # updated at explain time since user can override. Defines default LR schedule.
        self.max_iter = None

        # initialisation method and constraints for counterfactual
        self.solution_constraint = None
        if feature_range is not None:
            # TODO: test correctness
            self.solution_constraint = [feature_range[0], feature_range[1]]

        # used for user attribute setting validation
        self._expected_attributes = set()  # type: set
        # for convenience, to avoid if/else depending on framework in calling context
        self.device = None
        # below would need to be done for a PyTorchHelper with GPU support
        # subclasses provide PyTorch support, set by wrapper
        # self.to_numpy = partial(self.to_numpy, device=self.device)

        # a function used to slice predictor output so that it returns the target class output only.
        # Used by the calling context to slice predictor outputs
        self.cf_prediction_fcn = slice_prediction
        # fixes specified arguments of self.cf_prediction_fcn, set at explain time
        self._get_cf_prediction = None  # type: Union[None, Callable]

    def _set_predictor(self, predictor) -> None:
        """
        Sets the predictor as an attribute. In the case of `blakcbox` predictor, wraps it in a decorator
        which transforms input to `np.ndarray` and output to `tf.Tensor`.
        """
        if self.predictor_type == 'whitebox':
            self.predictor = predictor
        else:
            # wrap in a decorator that casts the input to np.ndarray and the output to tensor
            wrapper = get_blackbox_wrapper(self.framework)
            self.wrapper = wrapper
            self.predictor = wrapper(predictor)
            self.blackbox_eval_fcn = wachter_blackbox_wrapper

    def make_prediction(self, X) -> tf.Tensor:
        if self.predictor_type == 'whitebox':
            return self.predictor(X, training=False)
        else:
            return self.predictor(X)

    def check_constraint(self, cf: tf.Variable, target_proba: tf.Tensor, tol: float) -> bool:
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

        prediction = self.make_prediction(cf)
        return tf.reduce_all(
            tf.math.abs(self._get_cf_prediction(prediction, self.target_class) - target_proba) <= tol,
        ).numpy()

    def collect_step_data(self):
        """
        This function is used by the calling object in order to obtain the current state that is subsequently written to
        TensorBoard. As the state might require additional computation which is not necessary for steps where data is
        not logged, this method defers the functionality to `_get_current_state` where the state is computed.
        """
        return self.update_state()

    def update_state(self):
        """
        Computes quantities used solely for logging purposes. Called when data needs to be written to TensorBoard.
        If the logging requires expensive quantities that don't need to be computed at each iteration, they should be
        computed here and added to the state dictionary, which is collected by the calling object.
        """
        if self.predictor_type == 'whitebox':
            self.state['lr'] = self._get_learning_rate()
            return self.state

        # cannot call black-box predictor under the gradient tape, so the losses have to
        # be computed outside this context unlike the whitebox case
        model_output = self.make_prediction(self.solution)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        # require to re-evaluate the distance loss to account for the contrib of the numerical gradient
        dist_loss = self.distance_fcn(self.instance, self.solution)
        combined_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self.state['distance_loss'] = self.to_numpy(dist_loss).item()
        self.state['prediction_loss'] = self.to_numpy(pred_loss).item()
        self.state['total_loss'] = self.to_numpy(combined_loss).item()
        self.state['lr'] = self._get_learning_rate()

        return self.state

    def _get_learning_rate(self) -> tf.Tensor:
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """
        return self.optimizer._decayed_lr(tf.float32)

    def set_default_optimizer(self) -> None:
        """
        Initialises a default optimizer used for counterfactual search. This is the ADAM algorithm with linear learning
        rate decay from ``0.1`` to ``0.002`` over a number of iterations specified in
        `method_opts['search_opts']['max_iter']`. `method_opts` should be passed to the class where this object is
        instantiated.
        """

        self.lr_schedule = PolynomialDecay(0.1, self.max_iter, end_learning_rate=0.002, power=1.)
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer, optimizer_opts: Dict[str, Any]) -> None:
        """
        Sets the optimizer. If no optimizer is specified, a default optimizer is set.

        Parameters
        ----------
        optimizer
            An optimizer instance or optimizer class specified by user. Defaults to `None` if the user does not override
            the optimizer settings.
        optimizer_opts
            Options used to initialize the optimizer class.
        """

        # create a backup if the user does not override
        if optimizer is None:
            self.set_default_optimizer()
            # copy used for resetting optimizer for every lambda
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

    def step(self) -> None:
        """
        Runs a gradient descent step and updates current solution.
        """

        gradients = self.get_autodiff_gradients()
        self.apply_gradients(gradients)

    def get_autodiff_gradients(self) -> List[tf.Tensor]:
        """
        Calculates the gradients of the loss function with respect to the input (aka the counterfactual) at the current
        iteration.
        """

        with tf.GradientTape() as tape:
            loss = self.autograd_loss()
        gradients = tape.gradient(loss, [self.solution])

        return gradients

    def get_numerical_gradients(self) -> tf.Tensor:
        """"
        Differentiates the :math:`L_{pred}` part of the loss function. This is the product of a numerical 
        differentiation procedure and the gradient of :math:`L_{pred}` with respect to the model output, which is
        specified by the user as part of the loss specification.

        Returns
        -------
        A tensor of the same shape as the counterfactual searched representing the prediction function gradient wrt to 
        the predictor input.

        Notes
        ------
        The following assumptions are made:

            - `self.predictor` returns a scalar 
            - The shape of the prediction gradient is `self.cf.shape[0] (batch) x P (n_outputs) x self.cf.shape[1:] \
            (data point shape)`. 0-index slice below is due to the assumption that `self.predictor` returns a scalar. 
        """  # noqa W605
        if self.predictor_type == 'whitebox':
            return tf.zeros_like(self.solution)

        # shape of `prediction_gradient` is
        blackbox_wrap_fcn_args = (self.predictor, self.target_class, self.instance_class)
        prediction_gradient = self.num_grad_fcn(
            self.blackbox_eval_fcn,
            self.copy(self.solution),
            fcn_args=blackbox_wrap_fcn_args,
        )

        # see docstring to understand the slice
        prediction_gradient = prediction_gradient[:, 0, ...]
        pred = self.blackbox_eval_fcn(self.solution, self.predictor, self.target_class, self.instance_class)
        numerical_gradient = prediction_gradient * self.prediction_grad_fcn(pred, self.target_proba)

        assert numerical_gradient.shape == self.solution.shape

        return numerical_gradient

    def apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the gradients of the differentiable part of the loss.
        """

        # TODO: ALEX: TBD: SHOULD SELF.SOLUTION BE A LIST? I
        # TODO: ALEX: TBD: THIS MASK IS NOT TOO GENERIC? WE COULD IF ELSE ON IT?

        autograd_grads = gradients[0]
        numerical_grads = self.get_numerical_gradients()
        gradients = [self.mask * (autograd_grads + numerical_grads)]
        self.optimizer.apply_gradients(zip(gradients, [self.solution]))

    def reset_optimizer(self) -> None:
        """
        Resets the default optimizer. It is used to set the optimizer specified by the user as well as reset the 
        optimizer at each :math:`\lambda` optimisation cycle.

        Parametrs
        ---------
        optimizer
            This is either the opimizer passed by the user (first call) or a copy of it (during `_search`).
        """  # noqa W605

        self.optimizer = deepcopy(self._optimizer_copy)

    def initialise_variables(self, X: np.ndarray, optimized_features: np.ndarray,
                             **kwargs) -> None:  # type: ignore
        """
        Initialises optimisation variables so that the TensorFlow auto-differentiation framework can be used for
        counterfactual search.

        Parameters
        ----------
        X, optimized_features
            See calling object `counterfactual` method
        kwargs
            Valid kwargs are:
                - `target_class` Union[Literal['same', 'other'], int], `target_proba` (float), `instance_proba` (float), \
                `instance_class` (int)
            See calling object `counterfactual` method for more information.
        """  # noqa W605

        self.instance = tf.identity(X, name='instance')
        self.initialise_solution(X)
        self.mask = tf.identity(optimized_features, name='gradient mask')
        # tf.identity is the same as constant but does not always create tensors on CPU
        self.target_proba = tf.identity(kwargs.get('target_proba') * np.ones(1, dtype=X.dtype),
                                        name='target_proba')
        self.target_class = kwargs.get('target_class')
        self.instance_class = kwargs.get('instance_class')
        self.instance_proba = kwargs.get('instance_proba')

    def initialise_solution(self, X: np.ndarray) -> None:
        """
        Initializes the counterfactual to the data point `X` applies constraints on the feature value.

        Parameters
        ----------
        X
            Instance whose counterfactual is to be found.
        """

        constraint_fn = None
        if self.solution_constraint is not None:
            constraint_fn = partial(range_constraint, low=self.solution_constraint[0],
                                    high=self.solution_constraint[1])
        self.solution = tf.Variable(
            initial_value=X,
            trainable=True,
            name='counterfactual',
            constraint=constraint_fn,
        )

    def autograd_loss(self):
        """
        Concrete loss implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def to_numpy(X: Union[tf.Tensor, tf.Variable, np.ndarray]) -> np.ndarray:
        """
        Casts an array-like object tf.Tensor and tf.Variable objects to a `np.array` object.
        """

        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            return X.numpy()
        return X

    @staticmethod
    def to_tensor(X: np.ndarray) -> tf.Tensor:
        """
        Casts a numpy array object to tf.Tensor.
        """
        return tf.identity(X)

    @staticmethod
    def copy(X: tf.Variable) -> tf.Tensor:
        """
        Copies the value of the variable X into a new tensor
        """
        return tf.identity(X)


@register_backend(consumer_class='_WachterCounterfactual', predictor_type='whitebox')
class TFWachterOptimizerWB(TFGradientOptimizer):
    def __init__(self,
                 predictor,
                 loss_spec=None,
                 feature_range=None):
        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_WHITEBOX
        super().__init__(predictor=predictor,
                         loss_spec=loss_spec,
                         predictor_type='whitebox',
                         feature_range=feature_range)

    def autograd_loss(self) -> tf.Tensor:
        dist_loss = self.distance_fcn(self.instance, self.solution)
        model_output = self.make_prediction(self.solution)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        total_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        # updating state here to avoid extra evaluation later
        # TODO: this is wrong as the solution is not yet updated
        self.state['distance_loss'] = self.to_numpy(dist_loss).item()
        self.state['prediction_loss'] = self.to_numpy(pred_loss).item()
        self.state['total_loss'] = self.to_numpy(total_loss).item()

        return total_loss


@register_backend(consumer_class='_WachterCounterfactual', predictor_type='blackbox')
class TFWachterOptimizerBB(TFGradientOptimizer):
    def __init__(self,
                 predictor,
                 loss_spec=None,
                 feature_range=None):
        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_BLACKBOX
        super().__init__(predictor=predictor,
                         loss_spec=loss_spec,
                         predictor_type='blackbox',
                         feature_range=feature_range)

    def autograd_loss(self):
        dist_loss = self.distance_fcn(self.instance, self.solution)

        # multiply by self.lam here so that `apply_gradients` is generic
        return self.lam * dist_loss

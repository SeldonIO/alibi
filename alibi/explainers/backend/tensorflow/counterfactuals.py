import logging

import numpy as np
import tensorflow as tf

from alibi.utils.gradients import numerical_gradients
from collections import defaultdict
from copy import deepcopy
from functools import partial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from alibi.explainers.backend import register_backend


def scaled_l1_loss(instance: tf.Tensor, cf: tf.Variable, feature_scale: Optional[tf.Tensor] = None) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    # TODO: ALEX: TBD: USE TF BUILTINS

    ax_sum = tuple(np.arange(1, len(instance.shape)))
    if feature_scale is None:
        return tf.reduce_sum(tf.abs(cf - instance), axis=ax_sum, name='l1')
    return tf.reduce_sum(tf.abs(cf - instance) / feature_scale, axis=ax_sum, name='l1')


def squared_loss(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    # TODO: ALEX: TBD: THERE SHOULD BE A KERAS BUILT IN ?
    return tf.square(pred_probas - target_probas)


def squared_loss_grad(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING
    return 2*(pred_probas - target_probas)


def wachter_loss(distance: tf.Tensor, lam: float, pred: tf.Tensor) -> tf.Tensor:

    # TODO: DOCSTRING

    return lam * distance + pred


# TODO: CHECK TYPING CORRECTNESS FOR FUNCTIONS ABOVE

WACHTER_LOSS_SPEC_WHITEBOX = {
    'prediction': {'fcn': squared_loss, 'kwargs': {}},
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'loss': {'fcn': wachter_loss,  'kwargs': {}},  # function that combines the prediction and distance
}  # type: Dict[str, Mapping[str, Any]]
WACHTER_LOSS_SPEC_BLACKBOX = {
    'prediction': {
        'fcn': squared_loss,
        'kwargs': {},
        'grad_fcn': squared_loss_grad,
        'grad_fcn_kwargs': {},
        'gradient_method': {'name': 'central_difference', 'kwargs': {'eps': 0.01}}
    },
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'loss': {'fcn': wachter_loss, 'kwargs': {}},  # function that combines the prediction and distance
}  # type: Dict[str, Mapping[str, Any]]


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


def _slice_prediction(prediction: tf.Tensor, target_idx: Union[int, str], src_idx: int) -> tf.Tensor:
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
                             target_idx: Union[str, int],
                             src_idx: int) -> tf.Tensor:

    # TODO: DOCSTRING
    # TODO: WHEN WRITING DOCS, SPECIFY THE SHAPE THAT THE PREDICTOR IS EXPECTED TO TAKE IN AND RETURN
    prediction = predictor(X)
    pred_target = _slice_prediction(prediction, target_idx, src_idx)
    return pred_target


@register_backend(explainer_type='counterfactual', predictor_type='whitebox', method='wachter')
class TFWachterCounterfactualOptimizer:

    framework_name = 'tensorflow'

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        self.predictor = predictor
        self._expected_attributes = set(WACHTER_LOSS_SPEC_WHITEBOX)

        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_WHITEBOX

        # further loss spec properties (for black-box functions) are set in sub-classes
        for term in loss_spec:
            this_term_kwargs = loss_spec[term]['kwargs']  # type: ignore
            if this_term_kwargs:
                this_term_fcn = partial(loss_spec[term]['fcn'], **this_term_kwargs)
                self.__setattr__(f"{term}_fcn", this_term_fcn)
            else:
                self.__setattr__(f"{term}_fcn", loss_spec[term]['fcn'])

        self.state = defaultdict(lambda: None, dict.fromkeys(loss_spec.keys(), 0.0))
        self.state.update([('total_loss', 0.0)])

        # updated at explain time
        self.optimizer = None  # type: Union[tf.keras.optimizers.Optimizer, None]
        self._optimizer_copy = None
        # returns an index from the prediction that depends on the instance and counterfactual class
        self.cf_prediction_fcn = _slice_prediction
        # fixes specified arguments of self._cf_prediction, set at explain time
        self._get_cf_prediction = None  # type: Union[None, Callable]

        # problem variables: init. at explain time
        self.target_proba = None  # type: Union[tf.Tensor, None]
        self.instance = None  # type: Union[tf.Tensor, None]
        self.instance_class = None  # type: Union[int, None]
        self.cf = None  # type: Union[tf.Variable, None]
        # updated at explain time since user can override
        self.max_iter = None

        # initialisation method and constraints for counterfactual
        self.cf_constraint = None
        if feature_range is not None:
            self.cf_constraint = [feature_range[0], feature_range[1]]

        # updated by the optimisation outer loop
        self.lam = None  # type: Union[float, None]

        # for convenience, to avoid if/else in wrapper
        self.device = None
        # below would need to be done for a PyTorchHelper with GPU support
        # subclasses provide PyTorch support, set by wrapper
        # self.to_numpy_arr = partial(self.to_numpy_arr, device=self.device)

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)

    def __getattr__(self, key: Any) -> Any:
        return self.__getattribute__(key)

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

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer, optimizer_opts: Dict[str, Any]) -> None:

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

    def initialise_variables(self,
                             X: np.ndarray,
                             optimised_features: np.ndarray,
                             target_class: Union[int, str],
                             instance_class: int,
                             instance_proba: float,
                             target_proba: float) -> None:
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
        self.initialise_cf(X)
        self._mask = tf.identity(optimised_features, name='gradient mask')

    def cf_step(self, lam: float) -> None:
        """
        Runs a gradient descent step and updates current solution for a given value of `lam`.
        """

        self.lam = lam
        gradients = self.get_autodiff_gradients()
        self.apply_gradients(gradients)

    def get_autodiff_gradients(self) -> List[tf.Tensor]:
        """
        Calculates the gradients of the loss function with respect to the input (aka the counterfactual) at the current
        iteration.
        """

        with tf.GradientTape() as tape:
            loss = self.autograd_loss()
        gradients = tape.gradient(loss, [self.cf])

        return gradients

    def apply_gradients(self, gradients: List[tf.Tensor]) -> None:
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

    def autograd_loss(self) -> tf.Tensor:
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

        # TODO: DOCSTRING UPDATE

        dist_loss = self.distance_fcn(self.instance, self.cf)
        model_output = self.make_prediction(self.cf)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        total_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self.state['distance_loss'] = self.to_numpy_arr(dist_loss).item()
        self.state['prediction_loss'] = self.to_numpy_arr(pred_loss).item()
        self.state['total_loss'] = self.to_numpy_arr(total_loss).item()

        return total_loss

    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Makes a prediction for data points in `X`.

        Parameters
        ----------
        X
            A tensor or array for which predictions are requested.
        """

        return self.predictor(X, training=False)

    def initialise_cf(self, X: np.ndarray) -> None:
        """
        Initializes the counterfactual to the data point `X`.

        Parameters
        ----------
        X
            Instance whose counterfactual is to be found.
        """

        # TODO: UPDATE DOCSTRING
        constraint_fcn = None
        if self.cf_constraint is not None:
            constraint_fcn = partial(range_constraint, low=self.cf_constraint[0], high=self.cf_constraint[1])
        self.cf = tf.Variable(
            initial_value=X,
            trainable=True,
            name='counterfactual',
            constraint=constraint_fcn,
        )

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

    @staticmethod
    def to_numpy_arr(X: Union[tf.Tensor, tf.Variable, np.ndarray], **kwargs) -> np.ndarray:
        """
        Casts an array-like object tf.Tensor and tf.Variable objects to a `np.array` object.
        """

        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            return X.numpy()
        return X

    @staticmethod
    def to_tensor(X: np.ndarray, **kwargs) -> tf.Tensor:
        """
        Casts a numpy array object to tf.Tensor.
        """
        return tf.identity(X)

    @staticmethod
    def copy(X: tf.Variable, **kwargs) -> tf.Tensor:
        """
        Copies the value of the variable X into a new tensor
        """
        return tf.identity(X)

    def _get_current_state(self):

        # TODO: DOCSTRING
        # state precomputed as it just involves casting values to numpy arrays
        self.state['lr'] = self._get_learning_rate()
        return self.state

    def _get_learning_rate(self) -> tf.Tensor:
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """
        return self.optimizer._decayed_lr(tf.float32)

    def collect_step_data(self):

        # TODO: DOCSTRING
        return self._get_current_state()


@register_backend(explainer_type='counterfactual', predictor_type='blackbox', method='wachter')
class TFWachterCounterfactualOptimizerBB(TFWachterCounterfactualOptimizer):

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        super().__init__(predictor, loss_spec, feature_range, **kwargs)

        expected_attributes = set(WACHTER_LOSS_SPEC_BLACKBOX) | {'prediction_grad_fcn', '_num_grads_fn'}
        self._expected_attributes = expected_attributes
        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_BLACKBOX
        # TODO: ALEX: TBD: THIS PRACTICALLY MEANS ATTRS NOT IN THE SCHEMA WILL NOT BE DEFINED
        # set numerical gradients method and gradient fcns of nonlinear transformation wrt model output
        self._num_grads_fn = None  # type: Union[Callable, None]
        for term in loss_spec:
            if 'grad_fcn' in loss_spec[term]:
                this_term_grad_fcn_kwargs = loss_spec[term]['grad_fcn_kwargs']
                if this_term_grad_fcn_kwargs:
                    this_term_grad_fcn = partial(loss_spec[term]['grad_fcn'], this_term_grad_fcn_kwargs)
                    self.__setattr__(f"{term}_grad_fcn", this_term_grad_fcn)
                else:
                    self.__setattr__(f"{term}_grad_fcn", loss_spec[term]['grad_fcn'])
                available_grad_methods = [fcn for fcn in numerical_gradients[self.framework_name]]
                available_grad_methods_names = [fcn.__name__ for fcn in numerical_gradients[self.framework_name]]
                grad_method_name = loss_spec[term]['gradient_method']['name']
                grad_method_kwargs = loss_spec[term]['gradient_method']['kwargs']
                grad_method_idx = available_grad_methods_names.index(grad_method_name)
                if self._num_grads_fn is None:
                    if grad_method_kwargs:
                        self.__setattr__(
                            '_num_grads_fn', partial(available_grad_methods[grad_method_idx], **grad_method_kwargs)
                        )
                    else:
                        self.__setattr__('_num_grads_fn', available_grad_methods[grad_method_idx])
                else:
                    logging.warning(
                        f"Only one gradient computation method can be specified. Raise an issue if you wish to modify "
                        f"this behaviour. Method {loss_spec['term']['gradient_method']['name']} was ignored!"
                    )
        self.loss_spec = loss_spec
        self.blackbox_eval_fcn = wachter_blackbox_wrapper
        # wrap in a decorator that casts the input to np.ndarray and the output to tensor
        wrapper = kwargs.get("blackbox_wrapper")
        self.wrapper = wrapper
        self.predictor = wrapper(self.predictor)

    def autograd_loss(self):

        # TODO: ADD DOCSTRINGS

        dist_loss = self.distance_fcn(self.instance, self.cf)

        return dist_loss

    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        return self.predictor(X)

    def get_numerical_gradients(self) -> tf.Tensor:

        # TODO: DOCSTRING (INC. THE ASSUMPTION ON SELF.PREDICTOR TO RETURN SCALAR)
        # shape of `prediction_gradient` is self.cf.shape[0] (batch) x P (n_outputs) x self.cf.shape[1:]
        # (data point shape) 0-index slice below is due to the assumption that `self.predictor` returns a scalar

        blackbox_wrap_fcn_args = (self.predictor, self.target_class, self.instance_class)
        prediction_gradient = self._num_grads_fn(
            self.blackbox_eval_fcn,
            self.copy(self.cf),
            fcn_args=blackbox_wrap_fcn_args,
        )

        # see docstring to understand the slice
        prediction_gradient = prediction_gradient[:, 0, ...]
        pred = self.blackbox_eval_fcn(self.cf, self.predictor, self.target_class, self.instance_class)
        numerical_gradient = prediction_gradient * self.prediction_grad_fcn(pred, self.target_proba)

        assert numerical_gradient.shape == self.cf.shape

        return numerical_gradient

    def apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the gradients of the differentiable part of the loss.
        """

        autograd_grads = gradients[0]
        numerical_grads = self.get_numerical_gradients()
        gradients = [self._mask * (autograd_grads*self.lam + numerical_grads)]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def _get_current_state(self):

        # TODO: DOCSTRING
        # cannot call black-box predictor under the gradient tape, so the losses have to
        # be computed outside this context unlike the whitebox case
        model_output = self.make_prediction(self.cf)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        # require to re-evaluate the distance loss to account for the contrib of the numerical gradient
        dist_loss = self.distance_fcn(self.instance, self.cf)
        combined_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self.state['distance_loss'] = self.to_numpy_arr(dist_loss).item()
        self.state['prediction_loss'] = self.to_numpy_arr(pred_loss).item()
        self.state['total_loss'] = self.to_numpy_arr(combined_loss).item()
        self.state['lr'] = self._get_learning_rate()

        return self.state

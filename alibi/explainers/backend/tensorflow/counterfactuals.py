import logging

import numpy as np
import tensorflow as tf

from alibi.explainers.backend import register_backend
from alibi.utils.gradients import numerical_gradients
from collections import defaultdict
from copy import deepcopy
from functools import partial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, TYPE_CHECKING

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
    return 2*(pred_probas - target_probas)


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
    'prediction': {'fcn': squared_loss, 'kwargs': {}},
    'distance': {'fcn': scaled_l1_loss, 'kwargs': {'feature_scale': None}},
    'loss': {'fcn': wachter_loss,  'kwargs': {}},  # function that combines the prediction and distance
}  # type: Dict[str, Mapping[str, Any]]
"""
dict: A specification that allows customising the Wachter loss, defined as:

.. math:: L_{pred} + \lambda L_{dist}.

See our documentation for `details`_.

.. _details: https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html

This specification assumes that the search algorithm has access to the parameters of the predictor.

To specify :math:`L_{pred}`, the ``'prediction'`` field of the specification should have a callable under the ``'fcn'``
and its kwargs under the ``'kwargs'``. If there are no kwargs, an empty dictionary should be specified. The callable 
arguments are assumed to be the model prediction for the specified class and the target for the said prediction, 
specified as an input to `explain`, respectively.

Similarly, :math:`L_{dist}` is specified in the ``'distance'`` field of the specification. The args of the callable are
assumed to be :math:`X` the instance whose counterfactual is searched and the counterfactual at the current iteration 
:math:`X'`, respectively.

The ``'loss'`` field should be a function that combines the outputs of the prediction and distance parts (``'fcn'`` 
field) along with its kwargs.

"""
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
"""dict: 

A specification that allows customising the Wachter loss, defined as:

.. math:: L_{pred} + \lambda L_{dist}

See our `documeentation`_ for details.

.. _documentation: https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html


This specification assumes that the term specified in the ``'prediction'`` field cannot be differentiated using 
automatic differentiation frameworks. Consequently, this term requires additional information in the following fields: 

    - ``'gradient_method'``. The ``'name'`` sub-field specifies a function that can compute numerical gradients given \
    a black-box function and an input tensor :math:`X`. See `alibi.utils.tensorflow.gradients` for an example and \
    interface description and `alibi.explainers.backend.tensorflow.counterfactuals` `get_numerical_gradients` method for \
    an integration example. The kwargs of this function should be specified in the ``'kwargs'`` field

    - ``'grad_fcn'`` should be a callable taking the same inputs as ``'fcn'``. Represents the gradient of the prediction \
    term wrt model output. The output of this function is multiplied by the numerical gradient to give the gradient of \
    the non-diferentiable wrt to the input
   
    - ``'grad_fcn_kwargs'`` specify the kwargs of ``'grad_fcn'``. 
"""


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


def slice_prediction(prediction: tf.Tensor, target_idx: Union[int, str], src_idx: int) -> tf.Tensor:
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
                             target_idx: Union[str, int],
                             src_idx: int) -> tf.Tensor:

    """
    A wrapper that modifies a predictor that returns a vector output when queried with input :math:`X` predictor that
    returns a vector to return a scalar. The scalar depends on the `target_idx` and `src_idx`. The predictor (and the
    input) should be two-dimensional. Therefore, if a record of dimension :math:`D` is input, it should be passed as a
    tensor of shape `(1, D)`. A single-output classification model operating on this  instance should return a tensor
    or array with shape  `(1, 1)`.

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


class TFCounterfactualOptimizer:
    """A base class from which all optimizers used to search for counterfactuals should inherit. It implements basic
    functionality such as:

        - setting the loss terms as optimizer attributes
        - TensorFlow optimizer setter
        - basic variables initialisation
        - `cf_step` a method that differentiates the loss function specified and updates the solution
        - methods that copy or convert to/from tf.Tensor objects, that can be used by calling objects to implement \
        framework-independent functionality
        - a method that updates and returns the state dictionary to the calling object (used for logging purposes)
    """

    framework_name = 'tensorflow'

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Optional[Dict[str, Dict[str, Union[Callable, Dict]]]] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

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
        feature_range
            Used to constrain the range of the counterfactual features. It should be specified such that it can be
            compared element-wise with a counterfactual. See documentation for specific counterfactual implementations 
            (e.g., alibi.explainers.experimental.counterfactuals.WatcherCounterfactual) for more details about feature 
            range constraints.
        """  # noqa W605

        self.predictor = predictor

        # further loss spec properties (for black-box functions or more advanced functionality) are set in sub-classes
        for term in loss_spec:
            this_term_kwargs = loss_spec[term]['kwargs']  # type: ignore
            if this_term_kwargs:
                this_term_fcn = partial(loss_spec[term]['fcn'], **this_term_kwargs)
                self.__setattr__(f"{term}_fcn", this_term_fcn)
            else:
                self.__setattr__(f"{term}_fcn", loss_spec[term]['fcn'])

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
        self.cf = None  # type: Union[tf.Variable, None]
        # updated at explain time since user can override. Defines default LR schedule.
        self.max_iter = None

        # initialisation method and constraints for counterfactual
        self.cf_constraint = None
        if feature_range is not None:
            self.cf_constraint = [feature_range[0], feature_range[1]]

        # for convenience, to avoid if/else depending on framework in calling context
        self.device = None
        # below would need to be done for a PyTorchHelper with GPU support
        # subclasses provide PyTorch support, set by wrapper
        # self.to_numpy_arr = partial(self.to_numpy_arr, device=self.device)

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)

    def __getattr__(self, key: Any) -> Any:
        return self.__getattribute__(key)

    def set_default_optimizer(self) -> None:
        """
        Initialises a default optimizer used for counterfactual search. This is the ADAM algorithm with linear learning
        rate decay from ``0.1`` to ``0.002`` over a number of iterations specified in
        `method_opts['search_opts']['max_iter']`. `method_opts` should be passed to the class where this object is
        instantiated.
        """

        self.lr_schedule = PolynomialDecay(0.1, self.max_iter, end_learning_rate=0.002, power=1)
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

    def initialise_variables(self, X: np.ndarray, optimized_features: np.ndarray, **kwargs):
        """
        This method initializes:
            - a tensor that stores the value of the initial condition, `X`
            - a tensor that stores the counterfactual as the optimisation proceedes, `cf`
            - a tensor that tracks the optimized features, `mask`

        Sub-classes extending this method should:
            - call the super-class method to perform the aforementioned
            - pass the additional arguments via kwargs

        Parameters
        ----------
        X
            The instance whose counterfactual is to be searched.
        optimized_features
            A boolean mask, the same shape as `X`. A ``1`` indicates that a feature will be optimized whereas ``0``
            indicates the contrary
        """

        # tf.identity is the same as constant but does not always create tensors on CPU
        self.instance = tf.identity(X, name='instance')
        self.initialise_cf(X)
        self.mask = tf.identity(optimized_features, name='gradient mask')

    def initialise_cf(self, X: np.ndarray) -> None:
        """
        Initializes the counterfactual to the data point `X` applies constraints on the feature value.

        Parameters
        ----------
        X
            Instance whose counterfactual is to be found.
        """

        constraint_fcn = None
        if self.cf_constraint is not None:
            constraint_fcn = partial(range_constraint, low=self.cf_constraint[0], high=self.cf_constraint[1])
        self.cf = tf.Variable(
            initial_value=X,
            trainable=True,
            name='counterfactual',
            constraint=constraint_fcn,
        )

    def cf_step(self) -> None:
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
        gradients = tape.gradient(loss, [self.cf])

        return gradients

    def autograd_loss(self) -> tf.Tensor:
        """
        This function should be implemented by a sub-class in order to implement a specific loss function for
        counterfactual search. See 'TFWachterCounterfactualOptimizer' for a concrete implementation.
        """
        raise NotImplementedError("Loss should be implemented by sub-class!")

    def apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the loss.

        Parameters
        ----------
        gradients
            A list containing the gradients of the differentiable part of the loss.
        """

        autograd_grads = gradients[0]
        gradients = [self.mask * autograd_grads]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Makes a prediction for data points in `X`.

        Parameters
        ----------
        X
            A tensor or array for which predictions are requested.
        """

        return self.predictor(X, training=False)

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

    def _get_learning_rate(self) -> tf.Tensor:
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """
        return self.optimizer._decayed_lr(tf.float32)

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
        raise NotImplementedError("Sub-class should implemented method for updating state.")


@register_backend(explainer_type='counterfactual', predictor_type='whitebox', method='wachter')
class TFWachterCounterfactualOptimizer(TFCounterfactualOptimizer):
    """
    A TensorFlow helper class that differentiates the loss function:

        .. math:: \ell(X', X, \lambda) = L_{pred} + \lambda L_{dist}

    in the case where the optimizer has access to the parameters of the predictor ("whitebox" predictor type). The 
    differentiation is perfomed by the Tensorflow autograd library.
    """  # noqa W605

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        self._expected_attributes = set(WACHTER_LOSS_SPEC_WHITEBOX)

        # pass loss specification to the superclass
        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_WHITEBOX
        super().__init__(predictor, loss_spec, feature_range, **kwargs)

        # track the total loss
        self.state.update([('total_loss', 0.0)])

        # updated by the calling context in the optimisation loop
        self.lam = None  # type: Union[float, None]

        # TODO: ALEX: TBD: DESIGN ALTERNATIVES WRT TO WHAT PREDICTOR RETURNS
        # a function used to slice predictor output so that it returns the target class output only.
        # Used by the calling context to slice predictor outputs
        self.cf_prediction_fcn = slice_prediction
        # fixes specified arguments of self._cf_prediction, set at explain time
        self._get_cf_prediction = None  # type: Union[None, Callable]

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

    def initialise_variables(self, X: np.ndarray, optimized_features: np.ndarray, **kwargs) -> None:
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

        super().initialise_variables(X, optimized_features)
        # tf.identity is the same as constant but does not always create tensors on CPU
        self.target_proba = tf.identity(kwargs.get('target_proba') * np.ones(1, dtype=X.dtype), name='target_proba')
        self.target_class = kwargs.get('target_class')
        self.instance_class = kwargs.get('instance_class')
        self.instance_proba = kwargs.get('instance_proba')

    def autograd_loss(self) -> tf.Tensor:
        """
        Computes the loss specified above.

        The terms are defined as:
            - :math:`L_pred` is the callable specified in the ``'prediction'`` field of  `WACHTER_LOSS_SPEC_WHITEBOX`
            - :math:`L_dist` is the callable specified in the ``'distance'`` field of `WACHTER_LOSS_SPEC_WHITEBOX`
            - :math:`\ell(X, X' \lambda)` is the callable specified in the ``'loss'`` field of `WACHTER_LOSS_SPEC_WHITEBOX`

        These functions are set as optimizer attributes during initialization. This function assumes that the combined
        loss can be differentiated end-to-end (ie, access to model parameters).
`
        Returns
        -------
        A scalar representing the combined prediction and distance losses according to the function specified by the
        `loss_fcn` attribute.
        """   # noqa W605

        dist_loss = self.distance_fcn(self.instance, self.cf)
        model_output = self.make_prediction(self.cf)
        prediction = self._get_cf_prediction(model_output, self.target_class)
        pred_loss = self.prediction_fcn(prediction, self.target_proba)
        total_loss = self.loss_fcn(dist_loss, self.lam, pred_loss)

        self.state['distance_loss'] = self.to_numpy_arr(dist_loss).item()
        self.state['prediction_loss'] = self.to_numpy_arr(pred_loss).item()
        self.state['total_loss'] = self.to_numpy_arr(total_loss).item()

        return total_loss

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

    def update_state(self):
        """
        See superclass documentation.
        """
        # state precomputed as it just involves casting values to floats
        self.state['lr'] = self._get_learning_rate()
        return self.state


@register_backend(explainer_type='counterfactual', predictor_type='blackbox', method='wachter')
class TFWachterCounterfactualOptimizerBB(TFWachterCounterfactualOptimizer):
    """
    A TensorFlow helper class that differentiates the loss function:
    
        .. math:: \ell(X', X, \lambda) = L_{pred} + \lambda L_{dist}
    
    in the case where the optimizer does not have access to the parameters of the predictor ("blackbox" predictor type).
    The differentiation is approached as follows:
    
        - :math:`\lambda L_{dist}` is differentiated using TensorFlow autograd 
        - :math:`\lambda L_{pred}` is differentiated numerically. This requires the user to specify the derivative of \
        :math:`\lambda L_{pred}` wrt the model output in the `grad_fcn` field of the loss specification (see the default \
        specification `WACHTER_LOSS_SPEC_BLACKBOX` for an example)
        
    The gradients are applied to update the current solution. 
    """  # noqa W605

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):
        """
        Adds additional attributes to the class to support numerical differentiation of black-box predictors:

            - `num_grad_fcn`: this a numerical differentiation procedure that, given a callable, an input :math:`X`, a \
            set of keyword arguments and a set of additional arguments and keyword arguments for the callable, evaluates \
            the gradient of the callable with respect to the input :math:`X` 
            - `prediction_grad_fcn`: the derivative of :math:`L_{pred}` with respect to `pred`, the predictor output
        """  # noqa W605
        super().__init__(predictor, loss_spec, feature_range, **kwargs)

        # TODO: ALEX: TBD: THIS PRACTICALLY MEANS ATTRS NOT IN THE SCHEMA WILL NOT BE SET
        expected_attributes = set(WACHTER_LOSS_SPEC_BLACKBOX) | {'prediction_grad_fcn', '_num_grads_fn'}
        self._expected_attributes = expected_attributes
        if loss_spec is None:
            loss_spec = WACHTER_LOSS_SPEC_BLACKBOX
        # set numerical gradients method and gradient fcns of nonlinear transformation wrt model output
        self.num_grad_fcn = None  # type: Union[Callable, None]
        for term in loss_spec:
            if 'grad_fcn' in loss_spec[term]:
                this_term_grad_fcn_kwargs = loss_spec[term]['grad_fcn_kwargs']
                if this_term_grad_fcn_kwargs:
                    this_term_grad_fcn = partial(loss_spec[term]['grad_fcn'], this_term_grad_fcn_kwargs)
                    self.__setattr__(f"{term}_grad_fcn", this_term_grad_fcn)
                else:
                    self.__setattr__(f"{term}_grad_fcn", loss_spec[term]['grad_fcn'])
                available_grad_methods = [fcn for fcn in numerical_gradients[self.framework_name]]
                available_grad_methods_names = [fcn.__name__ for fcn in available_grad_methods]
                grad_method_name = loss_spec[term]['gradient_method']['name']
                grad_method_kwargs = loss_spec[term]['gradient_method']['kwargs']
                grad_method_idx = available_grad_methods_names.index(grad_method_name)
                if self.num_grad_fcn is None:
                    if grad_method_kwargs:
                        self.__setattr__(
                            'num_grad_fcn', partial(available_grad_methods[grad_method_idx], **grad_method_kwargs)
                        )
                    else:
                        self.__setattr__('num_grads_fcn', available_grad_methods[grad_method_idx])
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
        """
        Computes :math:`L_{dist}` part of the following loss function:


        Returns
        -------
        A scalar represting the value of :math:`L_{dist}`
        """
        dist_loss = self.distance_fcn(self.instance, self.cf)

        return dist_loss

    def make_prediction(self, X: Union[tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Queries the predictor on `X`.

        Returns
        -------
        A tensor representing the prediction.
        """
        return self.predictor(X)

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

        # shape of `prediction_gradient` is
        blackbox_wrap_fcn_args = (self.predictor, self.target_class, self.instance_class)
        prediction_gradient = self.num_grad_fcn(
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
        gradients = [self.mask * (autograd_grads*self.lam + numerical_grads)]
        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def update_state(self):
        """
        See superclass documentation.
        """
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

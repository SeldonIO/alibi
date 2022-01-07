import copy
import logging
import string
import warnings
from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf

from alibi.api.defaults import DEFAULT_DATA_INTGRAD, DEFAULT_META_INTGRAD
from alibi.api.interfaces import Explainer, Explanation
from alibi.utils.approximation_methods import approximation_parameters

logger = logging.getLogger(__name__)

_valid_output_shape_type: List = [tuple, list]


def _compute_convergence_delta(model: Union[tf.keras.models.Model],
                               input_dtypes: List[tf.DType],
                               attributions: List[np.ndarray],
                               start_point: Union[List[np.ndarray], np.ndarray],
                               end_point: Union[List[np.ndarray], np.ndarray],
                               forward_kwargs: Optional[dict],
                               target: Optional[List[int]],
                               _is_list: bool) -> np.ndarray:
    """
    Computes convergence deltas for each data point. Convergence delta measures how close the sum of all attributions
    is to the difference between the model output at the baseline and the model output at the data point.

    Parameters
    ----------
    model
        `tensorflow` model.
    input_dtypes
        List with data types of the inputs.
    attributions
        Attributions assigned by the integrated gradients method to each feature.
    start_point
        Baselines.
    end_point
        Data points.
    forward_kwargs
        Input keywords args.
    target
        Target for which the gradients are calculated for classification models.
    _is_list
        Whether the model's input is a `list` (multiple inputs) or a `np.narray` (single input).

    Returns
    -------
    Convergence deltas for each data point.
    """
    if forward_kwargs is None:
        forward_kwargs = {}
    if _is_list:
        start_point = [tf.convert_to_tensor(start_point[k], dtype=input_dtypes[k]) for k in range(len(input_dtypes))]
        end_point = [tf.convert_to_tensor(end_point[k], dtype=input_dtypes[k]) for k in range(len(input_dtypes))]
    else:
        start_point = tf.convert_to_tensor(start_point)
        end_point = tf.convert_to_tensor(end_point)

    def _sum_rows(inp):

        input_str = string.ascii_lowercase[1: len(inp.shape)]
        if isinstance(inp, tf.Tensor):
            sums = tf.einsum('a{}->a'.format(input_str), inp).numpy()
        elif isinstance(inp, np.ndarray):
            sums = np.einsum('a{}->a'.format(input_str), inp)
        else:
            raise NotImplementedError('input must be a tensorflow tensor or a numpy array')
        return sums

    start_out = _run_forward(model, start_point, target, forward_kwargs=forward_kwargs)
    end_out = _run_forward(model, end_point, target, forward_kwargs=forward_kwargs)

    if (len(model.output_shape) == 1 or model.output_shape[-1] == 1) and target is not None:
        target_tensor = tf.cast(target, dtype=start_out.dtype)
        target_tensor = tf.reshape(1 - target_tensor, [len(target), 1])
        sign = 2 * target_tensor - 1

        start_out = target_tensor + sign * start_out
        end_out = target_tensor + sign * end_out

    start_out_sum = _sum_rows(start_out)
    end_out_sum = _sum_rows(end_out)

    attr_sum = np.zeros(start_out_sum.shape)
    for j in range(len(attributions)):
        attrs_sum_j = _sum_rows(attributions[j])
        attr_sum += attrs_sum_j

    _deltas = attr_sum - (end_out_sum - start_out_sum)

    return _deltas


def _select_target(preds: tf.Tensor,
                   targets: Union[None, tf.Tensor, np.ndarray, list]) -> tf.Tensor:
    """
    Select the predictions corresponding to the targets if targets is not ``None``.

    Parameters
    ----------
    preds
        Predictions before selection.
    targets
        Targets to select.

    Returns
    -------
    Selected predictions.

    """
    if targets is not None:
        if isinstance(preds, tf.Tensor):
            preds = tf.linalg.diag_part(tf.gather(preds, targets, axis=1))
        else:
            raise NotImplementedError
    else:
        raise ValueError("target cannot be `None` if `model` output dimensions > 1")
    return preds


def _run_forward(model: Union[tf.keras.models.Model],
                 x: Union[List[tf.Tensor], List[np.ndarray], tf.Tensor, np.ndarray],
                 target: Union[None, tf.Tensor, np.ndarray, list],
                 forward_kwargs: Optional[dict] = None) -> tf.Tensor:
    """
    Returns the output of the model. If the target is not ``None``, only the output for the selected
    target is returned.

    Parameters
    ----------
    model
        `tensorflow` model.
    x
        Input data point.
    target
        Target for which the gradients are calculated for classification models.
    forward_kwargs
        Input keyword args.

    Returns
    -------
    Model output or model output after target selection for classification models.

    """
    if forward_kwargs is None:
        forward_kwargs = {}
    preds = model(x, **forward_kwargs)

    if len(model.output_shape) > 1 and model.output_shape[-1] > 1:
        preds = _select_target(preds, target)

    return preds


def _run_forward_from_layer(model: tf.keras.models.Model,
                            layer: tf.keras.layers.Layer,
                            orig_call: Callable,
                            orig_dummy_input: Union[list, np.ndarray],
                            x: tf.Tensor,
                            target: Union[None, tf.Tensor, np.ndarray, list],
                            forward_kwargs: Optional[dict] = None,
                            run_from_layer_inputs: bool = False,
                            select_target: bool = True) -> tf.Tensor:
    """
    Function currently unused.
    Executes a forward call from an internal layer of the model to the model output.

    Parameters
    ----------
    model
        `tensorflow` model.
    layer
        Starting layer for the forward call.
    orig_call
        Original `call` method of the layer.
    orig_dummy_input
        Dummy input needed to initiate the model forward call. The number of instances in the dummy input must
        be the same as the number of instances in `x`. The dummy input values play no role in the evaluation
        as the  layer's status is overwritten during the forward call.
    x
        Layer's inputs. The layer's status is overwritten with `x` during the forward call.
    target
        Target for the output position to be returned.
    forward_kwargs
        Input keyword args. It must be a dict with `numpy` arrays as values. If it's not ``None``,
        the first dimension of the arrays must correspond to the number of instances in `x` and orig_dummy_input.
    run_from_layer_inputs
        If ``True``, the forward pass starts from the layer's inputs, if ``False`` it starts from the layer's outputs.
    select_target
        Whether to return predictions for selected targets or return predictions for all targets.

    Returns
    -------
    Model's predictions for the given target.

    """

    def feed_layer(layer):
        """
        Overwrites the intermediate layer status with the precomputed values `x`.

        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Store the result and inputs of `layer.call` internally.
                if run_from_layer_inputs:
                    layer.inp = x
                    layer.result = func(*x, **kwargs)
                else:
                    layer.inp = args
                    layer.result = x
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)

    feed_layer(layer)
    if forward_kwargs is None:
        forward_kwargs = {}
    preds = model(orig_dummy_input, **forward_kwargs)

    delattr(layer, 'inp')
    delattr(layer, 'result')
    layer.call = orig_call

    if select_target and len(model.output_shape) > 1 and model.output_shape[-1] > 1:
        preds = _select_target(preds, target)

    return preds


def _run_forward_to_layer(model: tf.keras.models.Model,
                          layer: tf.keras.layers.Layer,
                          orig_call: Callable,
                          x: Union[List[np.ndarray], np.ndarray],
                          forward_kwargs: Optional[dict] = None,
                          run_to_layer_inputs: bool = False) -> tf.Tensor:
    """
    Executes a forward call from the model input to an internal layer output.

    Parameters
    ----------
    model
        `tensorflow` model.
    layer
        Starting layer for the forward call.
    orig_call
        Original `call` method of the layer.
    x
        Model's inputs.
    forward_kwargs
        Input keyword args.
    run_to_layer_inputs
        If ``True``, the layer's inputs are returned. If ``False``, the layer's output's are returned.

    Returns
    -------
    Output of the given layer.

    """
    if forward_kwargs is None:
        forward_kwargs = {}

    def take_layer(layer):
        """
        Stores the layer's outputs internally to the layer's object.

        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Store the result of `layer.call` internally.
                layer.inp = args
                layer.result = func(*args, **kwargs)
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)

    # inp = tf.zeros((x.shape[0], ) + model.input_shape[1:])
    take_layer(layer)
    _ = model(x, **forward_kwargs)
    layer_inp = layer.inp
    layer_out = layer.result

    delattr(layer, 'inp')
    delattr(layer, 'result')
    layer.call = orig_call

    if run_to_layer_inputs:
        return layer_inp
    else:
        return layer_out


def _forward_input_baseline(X: Union[List[np.ndarray], np.ndarray],
                            bls: Union[List[np.ndarray], np.ndarray],
                            model: tf.keras.Model,
                            layer: tf.keras.layers.Layer,
                            orig_call: Callable,
                            forward_kwargs: Optional[dict] = None,
                            forward_to_inputs: bool = False) -> Tuple[Union[list, tf.Tensor], Union[list, tf.Tensor]]:
    """
    Forwards inputs and baselines to the layer's inputs or outputs.

    Parameters
    ----------
    X
        Input data points.
    bls
        Baselines.
    model
        `tensorflow` model.
    layer
        Desired layer output.
    orig_call
        Original `call` method of layer.
    forward_kwargs
        Input keyword args.
    forward_to_inputs
        If ``True``, `X` and bls are forwarded to the layer's input. If ``False``, they are forwarded to
        the layer's outputs.

    Returns
    -------
    Forwarded inputs and baselines as a `numpy` arrays.

    """
    if forward_kwargs is None:
        forward_kwargs = {}
    if layer is not None:
        X_layer = _run_forward_to_layer(model,
                                        layer,
                                        orig_call,
                                        X,
                                        forward_kwargs=forward_kwargs,
                                        run_to_layer_inputs=forward_to_inputs)
        bls_layer = _run_forward_to_layer(model,
                                          layer,
                                          orig_call,
                                          bls,
                                          forward_kwargs=forward_kwargs,
                                          run_to_layer_inputs=forward_to_inputs)

        if isinstance(X_layer, tuple):
            X_layer = list(X_layer)

        if isinstance(bls_layer, tuple):
            bls_layer = list(bls_layer)

        return X_layer, bls_layer

    else:
        return X, bls


def _gradients_input(model: Union[tf.keras.models.Model],
                     x: List[tf.Tensor],
                     target: Union[None, tf.Tensor],
                     forward_kwargs: Optional[dict] = None) -> List[tf.Tensor]:
    """
    Calculates the gradients of the target class output (or the output if the output dimension is equal to 1)
    with respect to each input feature.

    Parameters
    ----------
    model
        `tensorflow` model.
    x
        Input data point.
    target
        Target for which the gradients are calculated if the output dimension is higher than 1.
    forward_kwargs
        Input keyword args.

    Returns
    -------
    Gradients for each input feature.

    """
    if forward_kwargs is None:
        forward_kwargs = {}
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = _run_forward(model, x, target, forward_kwargs=forward_kwargs)

    grads = tape.gradient(preds, x)

    return grads


def _gradients_layer(model: Union[tf.keras.models.Model],
                     layer: Union[tf.keras.layers.Layer],
                     orig_call: Callable,
                     orig_dummy_input: Union[list, np.ndarray],
                     x: Union[List[tf.Tensor], tf.Tensor],
                     target: Union[None, tf.Tensor],
                     forward_kwargs: Optional[dict] = None,
                     compute_layer_inputs_gradients: bool = False) -> tf.Tensor:
    """
    Calculates the gradients of the target class output (or the output if the output dimension is equal to 1)
    with respect to each element of `layer`.

    Parameters
    ----------
    model
        `tensorflow` model.
    layer
        Layer of the model with respect to which the gradients are calculated.
    orig_call
        Original `call` method of the layer. This is necessary since the call method is modified by the function
        in order to make the layer output visible to the `GradientTape`.
    x
        Input data point.
    target
        Target for which the gradients are calculated if the output dimension is higher than 1.
    forward_kwargs
        Input keyword args.
    compute_layer_inputs_gradients
        If ``True``, gradients are computed with respect to the layer's inputs.
        If ``False``, they are computed with respect to the layer's outputs.

    Returns
    -------
    Gradients for each element of layer.

    """

    def watch_layer(layer, tape):
        """
        Make an intermediate hidden `layer` watchable by the `tape`.
        After calling this function, you can obtain the gradient with
        respect to the output of the `layer` by calling:

            grads = tape.gradient(..., layer.result)

        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Store the result and the input of `layer.call` internally.
                if compute_layer_inputs_gradients:
                    layer.inp = x
                    layer.result = func(*x, **kwargs)
                    # From this point onwards, watch this tensor.
                    tape.watch(layer.inp)
                else:
                    layer.inp = args
                    layer.result = x
                    # From this point onwards, watch this tensor.
                    tape.watch(layer.result)
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)

    #  Repeating the dummy input needed to initiate the model's forward call in order to ensure that
    #  the number of dummy instances is the same as the number of real instances.
    #  This is necessary in case `forward_kwargs` is not None. In that case, the model forward call  would crash
    #  if the number of instances in `orig_dummy_input` is different from the number of instances in `forward_kwargs`.
    #  The number of instances in `forward_kwargs` is the same as the number of instances in `x` by construction.
    if isinstance(orig_dummy_input, list):
        if isinstance(x, list):
            orig_dummy_input = [np.repeat(inp, x[0].shape[0], axis=0) for inp in orig_dummy_input]
        else:
            orig_dummy_input = [np.repeat(inp, x.shape[0], axis=0) for inp in orig_dummy_input]
    else:
        if isinstance(x, list):
            orig_dummy_input = np.repeat(orig_dummy_input, x[0].shape[0], axis=0)
        else:
            orig_dummy_input = np.repeat(orig_dummy_input, x.shape[0], axis=0)

    if forward_kwargs is None:
        forward_kwargs = {}
    #  Calculating the gradients with respect to the layer.
    with tf.GradientTape() as tape:
        watch_layer(layer, tape)
        preds = _run_forward(model, orig_dummy_input, target, forward_kwargs=forward_kwargs)

    if compute_layer_inputs_gradients:
        grads = tape.gradient(preds, layer.inp)
    else:
        grads = tape.gradient(preds, layer.result)

    delattr(layer, 'inp')
    delattr(layer, 'result')
    layer.call = orig_call

    return grads


def _format_baseline(X: np.ndarray,
                     baselines: Union[None, int, float, np.ndarray]) -> np.ndarray:
    """
    Formats baselines to return a `numpy` array.

    Parameters
    ----------
    X
        Input data points.
    baselines
        Baselines.

    Returns
    -------
    Formatted inputs and  baselines as a `numpy` arrays.

    """
    if baselines is None:
        bls = np.zeros(X.shape).astype(X.dtype)
    elif isinstance(baselines, int) or isinstance(baselines, float):
        bls = np.full(X.shape, baselines).astype(X.dtype)
    elif isinstance(baselines, np.ndarray):
        bls = baselines.astype(X.dtype)
    else:
        raise ValueError(f"baselines must be `int`, `float`, `np.ndarray` or `None`. Found {type(baselines)}")

    return bls


def _format_target(target: Union[None, int, list, np.ndarray],
                   nb_samples: int) -> Union[None, List[int]]:
    """
    Formats target to return a list.

    Parameters
    ----------
    target
        Original target.
    nb_samples
        Number of samples in the batch.

    Returns
    -------
    Formatted target as a list.

    """
    if target is not None:
        if isinstance(target, int):
            target = [target for _ in range(nb_samples)]
        elif isinstance(target, list) or isinstance(target, np.ndarray):
            target = [t.astype(int) for t in target]
        else:
            raise NotImplementedError

    return target


def _get_target_from_target_fn(target_fn: Callable,
                               model: tf.keras.Model,
                               X: Union[np.ndarray, List[np.ndarray]],
                               forward_kwargs: Optional[dict] = None) -> np.ndarray:
    """
    Generate a target vector by using the `target_fn` to pick out a
    scalar dimension from the predictions.

    Parameters
    ----------
    target_fn
        Target function.
    model
        Model.
    X
        Data to be explained.
    forward_kwargs
        Any additional kwargs needed for the model forward pass.

    Returns
    -------
    Integer array of dimension `(N, )`.
    """
    if forward_kwargs is None:
        preds = model(X)
    else:
        preds = model(X, **forward_kwargs)

    # raise a warning if the predictions are scalar valued already
    # TODO: in the future we want to support outputs that are >2D at which point this check should change
    if preds.shape[-1] == 1:
        msg = "Predictions from the model are scalar valued but `target_fn` was passed. `target_fn` is not necessary" \
              "when predictions are scalar valued already. Using `target_fn` here may result in unexpected behaviour."
        warnings.warn(msg)

    target = target_fn(preds)
    expected_shape = (target.shape[0],)
    if target.shape != expected_shape:
        # TODO: in the future we want to support outputs that are >2D at which point this check should change
        msg = f"`target_fn` returned an array of shape {target.shape} but expected an array of shape {expected_shape}."
        raise ValueError(msg)  # TODO: raise a more specific error type?
    return target


def _sum_integral_terms(step_sizes: list,
                        grads: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    """
    Sums the terms in the path integral with weights `step_sizes`.

    Parameters
    ----------
    step_sizes
        Weights in the path integral sum.
    grads
        Gradients to sum for each feature.

    Returns
    -------
    Sums of the gradients along the chosen path.

    """
    input_str = string.ascii_lowercase[1: len(grads.shape)]
    if isinstance(grads, tf.Tensor):
        step_sizes = tf.convert_to_tensor(step_sizes)
        einstr = 'a,a{}->{}'.format(input_str, input_str)
        sums = tf.einsum(einstr, step_sizes, grads).numpy()
    elif isinstance(grads, np.ndarray):
        einstr = 'a,a{}->{}'.format(input_str, input_str)
        sums = np.einsum(einstr, step_sizes, grads)
    else:
        raise NotImplementedError('input must be a tensorflow tensor or a numpy array')
    return sums


def _calculate_sum_int(batches: List[List[tf.Tensor]],
                       model: Union[tf.keras.Model],
                       target: Union[None, List[int]],
                       target_paths: np.ndarray,
                       n_steps: int,
                       nb_samples: int,
                       step_sizes: List[float],
                       j: int) -> Union[tf.Tensor, np.ndarray]:
    """
    Calculates the sum of all the terms in the integral from a list of batch gradients.

    Parameters
    ----------
    batches
        List of batch gradients.
    model
        `tf.keras` or `keras` model.
    target
        List of targets.
    target_paths
        Targets for each path in the integral.
    n_steps
        Number of steps in the integral.
    nb_samples
        Total number of samples.
    step_sizes
        Step sizes used to calculate the integral.
    j
        Iterates through list of inputs or list of layers.

    Returns
    -------
    Sums of the gradients along the chosen path.
    """
    grads = tf.concat(batches[j], 0)
    shape = grads.shape[1:]
    if isinstance(shape, tf.TensorShape):
        shape = tuple(shape.as_list())

    # invert sign of gradients for target 0 examples if classifier returns only positive class probability
    if (len(model.output_shape) == 1 or model.output_shape[-1] == 1) and target is not None:
        sign = 2 * target_paths - 1
        grads = np.array([s * g for s, g in zip(sign, grads)])

    grads = tf.reshape(grads, (n_steps, nb_samples) + shape)
    # sum integral terms and scale attributions
    sum_int = _sum_integral_terms(step_sizes, grads.numpy())

    return sum_int


def _validate_output(model: tf.keras.Model,
                     target: Optional[List[int]]) -> None:
    """
    Validates the model's output type and raises an error if the output type is not supported.

    Parameters
    ----------
    model
        `Keras` model for which the output is validated.
    target
        Targets for which gradients are calculated
    """
    if not model.output_shape or not any(isinstance(model.output_shape, t) for t in _valid_output_shape_type):
        raise NotImplementedError(f"The model output_shape attribute must be in {_valid_output_shape_type}. "
                                  f"Found model.output_shape: {model.output_shape}")

    if (len(model.output_shape) == 1
        or model.output_shape[-1] == 1) \
            and target is None:
        logger.warning("It looks like you are passing a model with a scalar output and target is set to `None`."
                       "If your model is a regression model this will produce correct attributions. If your model "
                       "is a classification model, targets for each datapoint must be defined. "
                       "Not defining the target may lead to incorrect values for the attributions."
                       "Targets can be either the true classes or the classes predicted by the model.")


class IntegratedGradients(Explainer):

    def __init__(self,
                 model: tf.keras.Model,
                 layer: Optional[tf.keras.layers.Layer] = None,
                 target_fn: Optional[Callable] = None,
                 method: str = "gausslegendre",
                 n_steps: int = 50,
                 internal_batch_size: int = 100
                 ) -> None:
        """
        An implementation of the integrated gradients method for `tensorflow` models.

        For details of the method see the original paper: https://arxiv.org/abs/1703.01365 .

        Parameters
        ----------
        model
            `tensorflow` model.
        layer
            Layer with respect to which the gradients are calculated.
            If not provided, the gradients are calculated with respect to the input.
        method
            Method for the integral approximation. Methods available:
            ``"riemann_left"``, ``"riemann_right"``, ``"riemann_middle"``, ``"riemann_trapezoid"``, ``"gausslegendre"``.
        n_steps
            Number of step in the path integral approximation from the baseline to the input instance.
        internal_batch_size
            Batch size for the internal batching.
        """

        super().__init__(meta=copy.deepcopy(DEFAULT_META_INTGRAD))
        params = locals()
        remove = ['self', 'model', '__class__', 'layer']
        params = {k: v for k, v in params.items() if k not in remove}
        self.model = model

        if self.model.inputs is None:
            self._has_inputs = False
        else:
            self._has_inputs = True

        if layer is None:
            self.orig_call: Optional[Callable] = None
            layer_num: Optional[int] = 0
        else:
            self.orig_call = layer.call
            try:
                layer_num = model.layers.index(layer)
            except ValueError:
                logger.info("Layer not in the list of model.layers")
                layer_num = None

        params['layer'] = layer_num
        self.meta['params'].update(params)
        self.layer = layer
        self.n_steps = n_steps
        self.method = method
        self.internal_batch_size = internal_batch_size

        self._is_list: Optional[bool] = None
        self._is_np: Optional[bool] = None
        self.orig_dummy_input: Optional[Union[list, np.ndarray]] = None

        self.target_fn = target_fn

    def explain(self,
                X: Union[np.ndarray, List[np.ndarray]],
                forward_kwargs: Optional[dict] = None,
                baselines: Optional[Union[int, float, np.ndarray, List[int], List[float], List[np.ndarray]]] = None,
                target: Optional[Union[int, list, np.ndarray]] = None,
                attribute_to_layer_inputs: bool = False) -> Explanation:
        """Calculates the attributions for each input feature or element of layer and
        returns an Explanation object.

        Parameters
        ----------
        X
            Instance for which integrated gradients attribution are computed.
        forward_kwargs
            Input keyword args. If it's not ``None``, it must be a dict with `numpy` arrays as values.
            The first dimension of the arrays must correspond to the number of examples.
            It will be repeated for each of `n_steps` along the integrated path.
            The attributions are not computed with respect to these arguments.
        baselines
            Baselines (starting point of the path integral) for each instance.
            If the passed value is an `np.ndarray` must have the same shape as `X`.
            If not provided, all features values for the baselines are set to 0.
        target
            Defines which element of the model output is considered to compute the gradients.
            It can be a list of integers or a numeric value. If a numeric value is passed, the gradients are calculated
            for the same element of the output for all data points.
            It must be provided if the model output dimension is higher than 1.
            For regression models whose output is a scalar, target should not be provided.
            For classification models `target` can be either the true classes or the classes predicted by the model.
        attribute_to_layer_inputs
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If ``True``, gradients are computed for the layer's inputs, if ``False`` for the layer's outputs.

        Returns
        -------
        explanation
            `Explanation` object including `meta` and `data` attributes with integrated gradients attributions
            for each feature. See usage at `IG examples`_ for details.

            .. _IG examples:
                https://docs.seldon.io/projects/alibi/en/latest/methods/IntegratedGradients.html
        """
        # target handling logic
        if self.target_fn and target is not None:
            msg = 'Both `target_fn` and `target` were provided. Only one of these should be provided.'
            raise ValueError(msg)
        if self.target_fn:
            target = _get_target_from_target_fn(self.target_fn, self.model, X, forward_kwargs)

        self._is_list = isinstance(X, list)
        self._is_np = isinstance(X, np.ndarray)
        if forward_kwargs is None:
            forward_kwargs = {}

        if self._is_list:
            X = cast(List[np.ndarray], X)  # help mypy out
            self.orig_dummy_input = [np.zeros((1,) + xx.shape[1:], dtype=xx.dtype) for xx in X]  # type: ignore
            nb_samples = len(X[0])
            input_dtypes = [xx.dtype for xx in X]
            # Formatting baselines in case of models with multiple inputs
            if baselines is None:
                baselines = [None for _ in range(len(X))]  # type: ignore
            else:
                if not isinstance(baselines, list):
                    raise ValueError(f"If the input X is a list, baseline can only be `None` or "
                                     f"a list of the same length of X. Found baselines type {type(baselines)}")
                else:
                    if len(X) != len(baselines):
                        raise ValueError(f"Length of 'X' must match length of 'baselines'. "
                                         f"Found len(X): {len(X)}, len(baselines): {len(baselines)}")

            if max([len(x) for x in X]) != min([len(x) for x in X]):
                raise ValueError("First dimension must be egual for all inputs")

            for i in range(len(X)):
                x, baseline = X[i], baselines[i]  # type: ignore
                # format and check baselines
                baseline = _format_baseline(x, baseline)
                baselines[i] = baseline  # type: ignore

        elif self._is_np:
            X = cast(np.ndarray, X)  # help mypy out
            self.orig_dummy_input = np.zeros((1,) + X.shape[1:], dtype=X.dtype)  # type: ignore
            nb_samples = len(X)
            input_dtypes = [X.dtype]  # type: ignore
            # Formatting baselines for models with a single input
            baselines = _format_baseline(X, baselines)  # type: ignore # TODO: validate/narrow baselines type

        else:
            raise ValueError("Input must be a np.ndarray or a list of np.ndarray")

        # defining integral method
        step_sizes_func, alphas_func = approximation_parameters(self.method)
        step_sizes, alphas = step_sizes_func(self.n_steps), alphas_func(self.n_steps)
        target = _format_target(target, nb_samples)  # type: ignore[assignment]

        if self._is_list:
            X = cast(List[np.ndarray], X)  # help mypy out
            # Attributions calculation in case of multiple inputs
            if not self._has_inputs:
                # Inferring model's inputs from data points for models with no explicit inputs
                # (typically subclassed models)
                inputs = [tf.keras.Input(shape=xx.shape[1:], dtype=xx.dtype) for xx in X]
                self.model(inputs, **forward_kwargs)

            _validate_output(self.model, target)  # type: ignore[arg-type]

            if self.layer is None:
                # No layer passed, attributions computed with respect to the inputs
                attributions = self._compute_attributions_list_input(X,
                                                                     baselines,  # type: ignore[arg-type]
                                                                     target,
                                                                     step_sizes,
                                                                     alphas,
                                                                     nb_samples,
                                                                     forward_kwargs,
                                                                     attribute_to_layer_inputs)

            else:
                # forwad inputs and  baselines
                X_layer, baselines_layer = _forward_input_baseline(X,
                                                                   baselines,  # type: ignore[arg-type]
                                                                   self.model,
                                                                   self.layer,
                                                                   self.orig_call,  # type: ignore[arg-type]
                                                                   forward_kwargs=forward_kwargs,
                                                                   forward_to_inputs=attribute_to_layer_inputs)

                if isinstance(X_layer, list) and isinstance(baselines_layer, list):
                    attributions = self._compute_attributions_list_input(X_layer,
                                                                         baselines_layer,
                                                                         target,
                                                                         step_sizes,
                                                                         alphas,
                                                                         nb_samples,
                                                                         forward_kwargs,
                                                                         attribute_to_layer_inputs)
                else:
                    attributions = self._compute_attributions_tensor_input(X_layer,
                                                                           baselines_layer,
                                                                           target,
                                                                           step_sizes,
                                                                           alphas,
                                                                           nb_samples,
                                                                           forward_kwargs,
                                                                           attribute_to_layer_inputs)

        else:
            # Attributions calculation in case of single input
            if not self._has_inputs:
                inputs = tf.keras.Input(shape=X.shape[1:], dtype=X.dtype)  # type: ignore
                self.model(inputs, **forward_kwargs)

            _validate_output(self.model, target)

            if self.layer is None:
                attributions = self._compute_attributions_tensor_input(X,
                                                                       baselines,
                                                                       target,
                                                                       step_sizes,
                                                                       alphas,
                                                                       nb_samples,
                                                                       forward_kwargs,
                                                                       attribute_to_layer_inputs)

            else:
                # forwad inputs and  baselines
                X_layer, baselines_layer = _forward_input_baseline(X,
                                                                   baselines,  # type: ignore[arg-type]
                                                                   self.model,
                                                                   self.layer,
                                                                   self.orig_call,  # type: ignore[arg-type]
                                                                   forward_kwargs=forward_kwargs,
                                                                   forward_to_inputs=attribute_to_layer_inputs)

                if isinstance(X_layer, list) and isinstance(baselines_layer, list):
                    attributions = self._compute_attributions_list_input(X_layer,
                                                                         baselines_layer,
                                                                         target,
                                                                         step_sizes,
                                                                         alphas,
                                                                         nb_samples,
                                                                         forward_kwargs,
                                                                         attribute_to_layer_inputs)
                else:
                    attributions = self._compute_attributions_tensor_input(X_layer,
                                                                           baselines_layer,
                                                                           target,
                                                                           step_sizes,
                                                                           alphas,
                                                                           nb_samples,
                                                                           forward_kwargs,
                                                                           attribute_to_layer_inputs)
        # calculate convergence deltas
        deltas = _compute_convergence_delta(self.model,
                                            input_dtypes,
                                            attributions,
                                            baselines,  # type: ignore[arg-type]
                                            X,
                                            forward_kwargs,
                                            target,
                                            self._is_list)

        return self._build_explanation(
            X=X,
            forward_kwargs=forward_kwargs,
            baselines=baselines,  # type: ignore[arg-type]
            target=target,
            attributions=attributions,
            deltas=deltas
        )

    def _build_explanation(self,
                           X: Union[List[np.ndarray], np.ndarray],
                           forward_kwargs: Optional[dict],
                           baselines: List[np.ndarray],
                           target: Optional[List[int]],
                           attributions: Union[List[np.ndarray], List[tf.Tensor]],
                           deltas: np.ndarray) -> Explanation:
        if forward_kwargs is None:
            forward_kwargs = {}
        data = copy.deepcopy(DEFAULT_DATA_INTGRAD)
        predictions = self.model(X, **forward_kwargs).numpy()
        if isinstance(attributions[0], tf.Tensor):
            attributions = [attr.numpy() for attr in attributions]  # type: ignore[union-attr]
        data.update(X=X,
                    forward_kwargs=forward_kwargs,
                    baselines=baselines,
                    target=target,
                    attributions=attributions,
                    deltas=deltas,
                    predictions=predictions)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def reset_predictor(self, predictor: Union[tf.keras.Model]) -> None:
        """
        Resets the predictor model.

        Parameters
        ----------
        predictor
            New prediction model.
        """
        # TODO: check what else should be done (e.g. validate dtypes again?)
        self.model = predictor

    def _compute_attributions_list_input(self,
                                         X: List[np.ndarray],
                                         baselines: Union[List[int], List[float], List[np.ndarray]],
                                         target: Optional[List[int]],
                                         step_sizes: List[float],
                                         alphas: List[float],
                                         nb_samples: int,
                                         forward_kwargs: Optional[dict],
                                         compute_layer_inputs_gradients: bool) -> List:
        """For each tensor in a list of input tensors,
        calculates the attributions for each feature or element of layer.

        Parameters
        ----------
        X
            Instance for which integrated gradients attribution are computed.
        baselines
            Baselines (starting point of the path integral) for each instance.
        target
            Defines which element of the model output is considered to compute the gradients.
        step_sizes
            Weights in the path integral sum.
        alphas
            Interpolation parameter defining the points of the integral path.
        nb_samples
            Total number of samples.
        forward_kwargs
            Input keywords args.
        compute_layer_inputs_gradients
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If ``True``, gradients are computed for the layer's inputs, if ``False`` for the layer's outputs.

        Returns
        -------
        Tuple with integrated gradients attributions, deltas and predictions.

        """
        if forward_kwargs is None:
            forward_kwargs = {}
        attrs_dtypes = [xx.dtype for xx in X]

        # define paths in features' space
        paths = []
        for i in range(len(X)):
            x, baseline = X[i], baselines[i]  # type: ignore
            # construct paths
            path = np.concatenate([baseline + alphas[i] * (x - baseline) for i in range(self.n_steps)], axis=0)
            paths.append(path)

        if forward_kwargs:
            paths_kwargs = {k: np.concatenate([forward_kwargs[k] for _ in range(self.n_steps)], axis=0)
                            for k in forward_kwargs.keys()}  # type: Optional[dict]
        else:
            paths_kwargs = None

        # define target paths
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
        else:
            target_paths = None

        if forward_kwargs:
            if target_paths is not None:
                ds_args = tuple(p for p in paths) + (paths_kwargs, target_paths)
            else:
                ds_args = tuple(p for p in paths) + (paths_kwargs,)

        else:
            if target_paths is not None:
                ds_args = tuple(p for p in paths) + (target_paths,)
            else:
                ds_args = tuple(p for p in paths)

        paths_ds = tf.data.Dataset.from_tensor_slices(ds_args).batch(self.internal_batch_size)
        paths_ds.as_numpy_iterator()
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # calculate gradients for batches
        batches = []
        for path in paths_ds:
            if forward_kwargs:
                if target is not None:
                    paths_b, kwargs_b, target_b = path[:-2], path[-2], path[-1]
                else:
                    paths_b, kwargs_b = path
                    target_b = None
            else:
                if target is not None:
                    paths_b, target_b = path[:-1], path[-1]
                    kwargs_b = None
                else:
                    paths_b, kwargs_b, target_b = path, None, None

            paths_b = [tf.dtypes.cast(paths_b[i], attrs_dtypes[i]) for i in range(len(paths_b))]

            if self.layer is None:
                grads_b = _gradients_input(self.model, paths_b, target_b, forward_kwargs=kwargs_b)
            else:
                grads_b = _gradients_layer(self.model,
                                           self.layer,
                                           self.orig_call,  # type: ignore[arg-type]
                                           self.orig_dummy_input,  # type: ignore[arg-type]
                                           paths_b,
                                           target_b,
                                           forward_kwargs=kwargs_b,
                                           compute_layer_inputs_gradients=compute_layer_inputs_gradients)

            batches.append(grads_b)

        # multi-input
        batches = [[batches[i][j] for i in range(len(batches))] for j in range(len(attrs_dtypes))]

        # calculate attributions from gradients batches
        attributions = []
        for j in range(len(attrs_dtypes)):
            sum_int = _calculate_sum_int(batches, self.model,
                                         target, target_paths,
                                         self.n_steps, nb_samples,
                                         step_sizes, j)
            norm = X[j] - baselines[j]  # type: ignore
            attribution = norm * sum_int
            attributions.append(attribution)

        return attributions

    def _compute_attributions_tensor_input(self,
                                           X: Union[np.ndarray, tf.Tensor],
                                           baselines: Union[np.ndarray, tf.Tensor],
                                           target: Optional[List[int]],
                                           step_sizes: List[float],
                                           alphas: List[float],
                                           nb_samples: int,
                                           forward_kwargs: Optional[dict],
                                           compute_layer_inputs_gradients: bool) -> List:
        """For a single input tensor, calculates the attributions for each input feature or element of layer.

        Parameters
        ----------
        X
            Instance for which integrated gradients attribution are computed.
        baselines
            Baselines (starting point of the path integral) for each instance.
        target
            Defines which element of the model output is considered to compute the gradients.
        step_sizes
            Weights in the path integral sum.
        alphas
            Interpolation parameter defining the points of the integral path.
        nb_samples
            Total number of samples.
        forward_kwargs
            Inputs keywords args.
        compute_layer_inputs_gradients
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If ``True``, gradients are computed for the layer's inputs, if ``False`` for the layer's outputs.

        Returns
        -------
        Tuple with integrated gradients attributions, deltas and predictions.
        """
        if forward_kwargs is None:
            forward_kwargs = {}
        # define paths in features's or layers' space
        paths = np.concatenate([baselines + alphas[i] * (X - baselines) for i in range(self.n_steps)], axis=0)

        if forward_kwargs:
            paths_kwargs = {k: np.concatenate([forward_kwargs[k] for _ in range(self.n_steps)], axis=0)
                            for k in forward_kwargs.keys()}  # type: Optional[dict]
        else:
            paths_kwargs = None

        # define target paths
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
        else:
            target_paths = None

        if forward_kwargs:
            if target_paths is not None:
                ds_args = (paths, paths_kwargs, target_paths)
            else:
                ds_args = (paths, paths_kwargs)  # type: ignore
        else:
            if target_paths is not None:
                ds_args = (paths, target_paths)  # type: ignore
            else:
                ds_args = paths

        paths_ds = tf.data.Dataset.from_tensor_slices(ds_args).batch(self.internal_batch_size)
        paths_ds.as_numpy_iterator()
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # calculate gradients for batches
        batches = []
        for path in paths_ds:
            if forward_kwargs:
                if target is not None:
                    paths_b, kwargs_b, target_b = path
                else:
                    paths_b, kwargs_b = path
                    target_b = None
            else:
                kwargs_b = None
                if target is not None:
                    paths_b, target_b = path
                else:
                    paths_b, target_b = path, None

            if self.layer is None:
                grads_b = _gradients_input(self.model, paths_b, target_b, forward_kwargs=kwargs_b)

            else:
                grads_b = _gradients_layer(self.model,
                                           self.layer,
                                           self.orig_call,  # type: ignore[arg-type]
                                           self.orig_dummy_input,  # type: ignore[arg-type]
                                           paths_b,
                                           target_b,
                                           forward_kwargs=kwargs_b,
                                           compute_layer_inputs_gradients=compute_layer_inputs_gradients)

            batches.append(grads_b)

        # calculate attributions from gradients batches
        attributions = []
        sum_int = _calculate_sum_int([batches], self.model,
                                     target, target_paths,
                                     self.n_steps, nb_samples,
                                     step_sizes, 0)
        norm = X - baselines

        attribution = norm * sum_int
        attributions.append(attribution)

        return attributions

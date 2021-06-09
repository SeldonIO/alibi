import copy
import logging
import numpy as np
import string
import tensorflow as tf

from alibi.api.defaults import DEFAULT_DATA_INTGRAD, DEFAULT_META_INTGRAD
from alibi.utils.approximation_methods import approximation_parameters
from alibi.api.interfaces import Explainer, Explanation
from typing import Callable, Union, List, Tuple, Optional

logger = logging.getLogger(__name__)


def _compute_convergence_delta(model: Union[tf.keras.models.Model],
                               layer: Union[tf.keras.layers.Layer],
                               orig_dummy_input: Union[list, np.ndarray],
                               input_dtypes: List[tf.DType],
                               attributions: List[np.ndarray],
                               start_point: Union[List[np.ndarray], np.ndarray],
                               end_point: Union[List[np.ndarray], np.ndarray],
                               target: Optional[List[int]],
                               _is_list: bool,
                               layer_inputs_attributions: bool) -> np.ndarray:
    """
    Computes convergence deltas for each data point. Convergence delta measures how close the sum of all attributions
    is to the difference between the model output at the baseline and the model output at the data point.

    Parameters
    ----------
    model
        Tensorflow or keras model.
    layer
        Layer for which attributions are computed.
        If None, attributions are assumed to be computed with respect to the model inputs.
    orig_dummy_input
        Dummy input needed to initiate the model forward call when start_point, end_point and attributions refer to
        an internal layer. The correct values layer input are overwritten during the forward call.
    input_dtypes
        List with data types of the inputs.
    attributions
        Attributions assigned by the integrated gradients method to each feature.
    start_point
        Baselines.
    end_point
        Data points.
    target
        Target for which the gradients are calculated for classification models.
    _is_list
        Whether the model's input is a list (multiple inputs) or a np array (single input).
    layer_inputs_attributions
        Whether the attributions refer to the layer's inputs or output in case of layer's attributions.
        If True, it is assumed that the attributions refer to the inputs, if False that they refer to the outputs.

    Returns
    -------
        Convergence deltas for each data point.
    """
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

    if layer is not None:
        orig_call = layer.call
        start_out = _run_forward_from_layer(model,
                                            layer,
                                            orig_call,
                                            orig_dummy_input,
                                            start_point,
                                            target,
                                            run_from_layer_inputs=layer_inputs_attributions)
        end_out = _run_forward_from_layer(model,
                                          layer,
                                          orig_call,
                                          orig_dummy_input,
                                          end_point,
                                          target,
                                          run_from_layer_inputs=layer_inputs_attributions)
    else:
        start_out = _run_forward(model, start_point, target)
        end_out = _run_forward(model, end_point, target)

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


def _select_target(ps, ts):
    if ts is not None:
        if isinstance(ps, tf.Tensor):
            ps = tf.linalg.diag_part(tf.gather(ps, ts, axis=1))
        else:
            raise NotImplementedError
    else:
        raise ValueError("target cannot be `None` if `model` output dimensions > 1")
    return ps


def _run_forward(model: Union[tf.keras.models.Model],
                 x: Union[List[tf.Tensor], List[np.ndarray]],
                 target: Union[None, tf.Tensor, np.ndarray, list]) -> tf.Tensor:
    """
    Returns the output of the model. If the target is not `None`, only the output for the selected target is returned.

    Parameters
    ----------
    model
        Tensorflow or keras model.
    x
        Input data point.
    target
        Target for which the gradients are calculated for classification models.

    Returns
    -------
        Model output or model output after target selection for classification models.

    """
    preds = model(x)
    if len(model.output_shape) > 1 and model.output_shape[-1] > 1:
        preds = _select_target(preds, target)

    return preds


def _run_forward_from_layer(model: tf.keras.models.Model,
                            layer: tf.keras.layers.Layer,
                            orig_call: Callable,
                            orig_dummy_input: Union[list, np.ndarray],
                            x: tf.Tensor,
                            target: Union[None, tf.Tensor, np.ndarray, list],
                            run_from_layer_inputs: bool = False) -> tf.Tensor:
    """
    Executes a forward call from an internal layer of the model to the model output.
    Parameters
    ----------
    model
        Tensorflow or keras model.
    layer
        Starting layer for the forward call.
    orig_call
        Original `call` method of the layer.
    orig_dummy_input
        Dummy input needed to initiate the model forward call.
        The  layer's status is overwritten during the forward call.
    x
        Layer's inputs. The layer's status is overwritten with `x` during the forward call.
    target
        Target for the output position to be returned.
    run_from_layer_inputs
        If True, the forward pass starts from the layer's inputs, if False it starts from the layer's outputs.

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
        return layer

    feed_layer(layer)
    preds = model(orig_dummy_input)

    delattr(layer, 'inp')
    delattr(layer, 'result')
    layer.call = orig_call

    if len(model.output_shape) > 1 and model.output_shape[-1] > 1:
        preds = _select_target(preds, target)

    return preds


def _run_forward_to_layer(model: tf.keras.models.Model,
                          layer: tf.keras.layers.Layer,
                          orig_call: Callable,
                          x: Union[List[np.ndarray], np.ndarray],
                          run_to_layer_inputs: bool = False) -> tf.Tensor:
    """
    Executes a forward call from the model input to an internal layer output.
    Parameters
    ----------
    model
        Tensorflow or keras model.
    layer
        Starting layer for the forward call.
    orig_call
        Original `call` method of the layer.
    x
        Model's inputs.
    run_to_layer_inputs
        If True, the layer's inputs are returned. If False, the layer's output's are returned.

    Returns
    -------
        Output of the given layer.

    """
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
        return layer

    # inp = tf.zeros((x.shape[0], ) + model.input_shape[1:])
    take_layer(layer)
    _ = model(x)
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
                            forward_to_inputs: bool = False) -> Tuple[Union[list, tf.Tensor], Union[list, tf.Tensor]]:
    """
    Forwards inputs and baselines to the output layer of `model_to_layer`.

    Parameters
    ----------
    X
        Input data points.
    bls
        Baselines.
    model
        Tensorflow or keras model.
    layer
        Desired layer output.
    orig_call
        Original `call` method of layer.
    forward_to_inputs
        If True, X and bls are forwarded to the layer's input. If False, they are forwarded to the layer's outputs.

    Returns
    -------
        Forwarded inputs and  baselines as a numpy arrays.

    """
    if layer is not None:
        X_layer = _run_forward_to_layer(model, layer, orig_call, X, run_to_layer_inputs=forward_to_inputs)
        bls_layer = _run_forward_to_layer(model, layer, orig_call, bls, run_to_layer_inputs=forward_to_inputs)

        if isinstance(X_layer, tuple):
            X_layer = list(X_layer)

        if isinstance(bls_layer, tuple):
            bls_layer = list(bls_layer)

        return X_layer, bls_layer

    else:
        return X, bls


def _gradients_input(model: Union[tf.keras.models.Model],
                     x: List[tf.Tensor],
                     target: Union[None, tf.Tensor]) -> List[tf.Tensor]:
    """
    Calculates the gradients of the target class output (or the output if the output dimension is equal to 1)
    with respect to each input feature.

    Parameters
    ----------
    model
        Tensorflow or keras model.
    x
        Input data point.
    target
        Target for which the gradients are calculated if the output dimension is higher than 1.

    Returns
    -------
        Gradients for each input feature.

    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = _run_forward(model, x, target)

    grads = tape.gradient(preds, x)

    return grads


def _gradients_layer(model: Union[tf.keras.models.Model],
                     layer: Union[tf.keras.layers.Layer],
                     orig_call: Callable,
                     orig_dummy_input: Union[list, np.ndarray],
                     x: Union[List[tf.Tensor], tf.Tensor],
                     target: Union[None, tf.Tensor],
                     compute_layer_inputs_gradients: bool = False) -> tf.Tensor:
    """
    Calculates the gradients of the target class output (or the output if the output dimension is equal to 1)
    with respect to each element of `layer`.

    Parameters
    ----------
    model
        Tensorflow or keras model.
    layer
        Layer of the model with respect to which the gradients are calculated.
    orig_call
        Original `call` method of the layer. This is necessary since the call method is modified by the function
        in order to make the layer output visible to the GradientTape.
    x
        Input data point.
    target
        Target for which the gradients are calculated if the output dimension is higher than 1.
    compute_layer_inputs_gradients
        If True, gradients are computed respect to the layer's inputs.
        If False, they are computed respect to the layer's outputs.

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
                else:
                    layer.inp = args
                    layer.result = x
                # From this point onwards, watch this tensors.
                tape.watch(layer.inp)
                tape.watch(layer.result)
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)
        return layer

    with tf.GradientTape() as tape:
        watch_layer(layer, tape)
        preds = _run_forward(model, orig_dummy_input, target)

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
    Formats baselines to return a numpy array.

    Parameters
    ----------
    X
        Input data points.
    baselines
        Baselines.

    Returns
    -------
        Formatted inputs and  baselines as a numpy arrays.

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
        tf.keras or keras model.
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


class IntegratedGradients(Explainer):

    def __init__(self,
                 model: tf.keras.Model,
                 layer: Optional[tf.keras.layers.Layer] = None,
                 method: str = "gausslegendre",
                 n_steps: int = 50,
                 internal_batch_size: int = 100
                 ) -> None:
        """
        An implementation of the integrated gradients method for Tensorflow and Keras models.

        For details of the method see the original paper:
        https://arxiv.org/abs/1703.01365 .

        Parameters
        ----------
        model
            Tensorflow or Keras model.
        layer
            Layer with respect to which the gradients are calculated.
            If not provided, the gradients are calculated with respect to the input.
        method
            Method for the integral approximation. Methods available:
            "riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid", "gausslegendre".
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
            self.orig_call = None
            layer_num = 0
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

    def explain(self,
                X: Union[np.ndarray, List[np.ndarray]],
                baselines: Union[int, float, np.ndarray, List[int], List[float], List[np.ndarray]] = None,
                target: Union[int, list, np.ndarray] = None,
                compute_layer_inputs_gradients: bool = False) -> Explanation:
        """Calculates the attributions for each input feature or element of layer and
        returns an Explanation object.

        Parameters
        ----------
        X
            Instance for which integrated gradients attribution are computed.
        baselines
            Baselines (starting point of the path integral) for each instance.
            If the passed value is an `np.ndarray` must have the same shape as X.
            If not provided, all features values for the baselines are set to 0.
        target
            Defines which element of the model output is considered to compute the gradients.
            It can be a list of integers or a numeric value. If a numeric value is passed, the gradients are calculated
            for the same element of the output for all data points.
            It must be provided if the model output dimension is higher than 1.
            For regression models whose output is a scalar, target should not be provided.
            For classification models `target` can be either the true classes or the classes predicted by the model.
        compute_layer_inputs_gradients
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If True, gradients are computed for the layer's inputs, if False for the layer's outputs.

        Returns
        -------
            `Explanation` object including `meta` and `data` attributes with integrated gradients attributions
            for each feature.

        """
        self._is_list = isinstance(X, list)
        self._is_np = isinstance(X, np.ndarray)

        if self._is_list:
            self.orig_dummy_input = [np.zeros((1,) + xx.shape[1:], dtype=xx.dtype) for xx in X]  # type: ignore
            nb_samples = len(X[0])
            # Formatting baselines in case of models with multiple inputs
            if baselines is None:
                baselines = [None for _ in range(len(X))]
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
            self.orig_dummy_input = np.zeros((1,) + X.shape[1:], dtype=X.dtype)  # type: ignore
            nb_samples = len(X)
            # Formatting baselines for models with a single input
            baselines = _format_baseline(X, baselines)

        else:
            raise ValueError("Input must be a np.ndarray or a list of np.ndarray")

        # defining integral method
        step_sizes_func, alphas_func = approximation_parameters(self.method)
        step_sizes, alphas = step_sizes_func(self.n_steps), alphas_func(self.n_steps)
        target = _format_target(target, nb_samples)

        if self._is_list:
            # Attributions calculation in case of multiple inputs
            if not self._has_inputs:
                # Inferring model's inputs from data points for models with no explicit inputs
                # (typically subclassed models)
                inputs = [tf.keras.Input(shape=xx.shape[1:], dtype=xx.dtype) for xx in X]
                self.model(inputs)

            if self.layer is None:
                # No layer passed, attributions computed with respect to the inputs
                attributions, deltas = self._compute_attributions_list_input(X,
                                                                             baselines,
                                                                             target,
                                                                             step_sizes,
                                                                             alphas,
                                                                             nb_samples,
                                                                             compute_layer_inputs_gradients)

            else:
                # forwad inputs and  baselines
                X_layer, baselines_layer = _forward_input_baseline(X,
                                                                   baselines,
                                                                   self.model,
                                                                   self.layer,
                                                                   self.orig_call,
                                                                   forward_to_inputs=compute_layer_inputs_gradients)

                if isinstance(X_layer, list) and isinstance(baselines_layer, list):
                    attributions, deltas = self._compute_attributions_list_input(X_layer,
                                                                                 baselines_layer,
                                                                                 target,
                                                                                 step_sizes,
                                                                                 alphas,
                                                                                 nb_samples,
                                                                                 compute_layer_inputs_gradients)
                else:
                    attributions, deltas = self._compute_attributions_tensor_input(X_layer,
                                                                                   baselines_layer,
                                                                                   target,
                                                                                   step_sizes,
                                                                                   alphas,
                                                                                   nb_samples,
                                                                                   compute_layer_inputs_gradients)

        else:
            # Attributions calculation in case of single input
            if not self._has_inputs:
                inputs = tf.keras.Input(shape=X.shape[1:], dtype=X.dtype)  # type: ignore
                self.model(inputs)

            if self.layer is None:
                attributions, deltas = self._compute_attributions_tensor_input(X,
                                                                               baselines,
                                                                               target,
                                                                               step_sizes,
                                                                               alphas,
                                                                               nb_samples,
                                                                               compute_layer_inputs_gradients)

            else:
                # forwad inputs and  baselines
                X_layer, baselines_layer = _forward_input_baseline(X,
                                                                   baselines,
                                                                   self.model,
                                                                   self.layer,
                                                                   self.orig_call,
                                                                   forward_to_inputs=compute_layer_inputs_gradients)

                if isinstance(X_layer, list) and isinstance(baselines_layer, list):
                    attributions, deltas = self._compute_attributions_list_input(X_layer,
                                                                                 baselines_layer,
                                                                                 target,
                                                                                 step_sizes,
                                                                                 alphas,
                                                                                 nb_samples,
                                                                                 compute_layer_inputs_gradients)
                else:
                    attributions, deltas = self._compute_attributions_tensor_input(X_layer,
                                                                                   baselines_layer,
                                                                                   target,
                                                                                   step_sizes,
                                                                                   alphas,
                                                                                   nb_samples,
                                                                                   compute_layer_inputs_gradients)

        return self.build_explanation(
            X=X,
            baselines=baselines,
            target=target,
            attributions=attributions,
            deltas=deltas
        )

    def build_explanation(self,
                          X: List[np.ndarray],
                          baselines: List[np.ndarray],
                          target: Optional[List[int]],
                          attributions: Union[List[np.ndarray], List[tf.Tensor]],
                          deltas: np.ndarray) -> Explanation:

        data = copy.deepcopy(DEFAULT_DATA_INTGRAD)
        predictions = self.model(X).numpy()
        if isinstance(attributions[0], tf.Tensor):
            attributions = [attr.numpy() for attr in attributions]
        data.update(X=X,
                    baselines=baselines,
                    target=target,
                    attributions=attributions,
                    deltas=deltas,
                    predictions=predictions)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def reset_predictor(self, predictor: Union[tf.keras.Model]) -> None:
        # TODO: check what else should be done (e.g. validate dtypes again?)
        self.model = predictor

    def _compute_attributions_list_input(self,
                                         X: List[np.ndarray],
                                         baselines: Union[List[int], List[float], List[np.ndarray]],
                                         target: Optional[List[int]],
                                         step_sizes: List[float],
                                         alphas: List[float],
                                         nb_samples: int,
                                         compute_layer_inputs_gradients: bool) -> Tuple:
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
            Interpolation parameter defining the points of the interal path.
        nb_samples
            Total number of samples.
        compute_layer_inputs_gradients
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If True, gradients are computed for the layer's inputs, if False for the layer's outputs.

        Returns
        -------
            Tuple with integrated gradients attributions, deltas and predictions

        """
        input_dtypes = [xx.dtype for xx in X]

        if (len(self.model.output_shape) == 1
            or self.model.output_shape[-1] == 1) \
                and target is None:
            logger.warning("It looks like you are passing a model with a scalar output and target is set to `None`."
                           "If your model is a regression model this will produce correct attributions. If your model "
                           "is a classification model, targets for each datapoint must be defined. "
                           "Not defining the target may lead to incorrect values for the attributions."
                           "Targets can be either the true classes or the classes predicted by the model.")

        # define paths in features' space
        paths = []
        for i in range(len(X)):
            x, baseline = X[i], baselines[i]  # type: ignore
            # construct paths
            path = np.concatenate([baseline + alphas[i] * (x - baseline) for i in range(self.n_steps)], axis=0)
            paths.append(path)

        # define target paths
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
        else:
            target_paths = None

        def generator(target_paths=target_paths):
            """Generates paths - targets pairs"""
            if target_paths is not None:
                inps_labels = paths + [target_paths]
                for y in zip(*inps_labels):
                    yield tuple(y[i] for i in range(len(y) - 1)), y[-1]
            else:
                for y in zip(*paths):
                    yield y

        if target is not None:
            paths_ds = tf.data.Dataset.from_generator(generator,
                                                      output_types=(tuple(input_dtypes),
                                                                    tf.int64)).batch(self.internal_batch_size)
        else:
            paths_ds = tf.data.Dataset.from_generator(generator, output_types=tuple(input_dtypes)).batch(
                self.internal_batch_size)

        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # calculate gradients for batches
        batches = []
        for path in paths_ds:

            if target is not None:
                paths_b, target_b = path
            else:
                paths_b, target_b = path, None

            paths_b = [tf.dtypes.cast(paths_b[i], input_dtypes[i]) for i in range(len(paths_b))]

            if self.layer is None:
                grads_b = _gradients_input(self.model, paths_b, target_b)
            else:
                grads_b = _gradients_layer(self.model,
                                           self.layer,
                                           self.orig_call,
                                           self.orig_dummy_input,
                                           paths_b,
                                           target_b,
                                           compute_layer_inputs_gradients=compute_layer_inputs_gradients)

            batches.append(grads_b)

        # multi-input
        batches = [[batches[i][j] for i in range(len(batches))] for j in range(len(input_dtypes))]

        # calculate attributions from gradients batches
        attributions = []
        for j in range(len(input_dtypes)):
            sum_int = _calculate_sum_int(batches, self.model,
                                         target, target_paths,
                                         self.n_steps, nb_samples,
                                         step_sizes, j)
            norm = X[j] - baselines[j]  # type: ignore
            attribution = norm * sum_int
            attributions.append(attribution)

        # calculate convergence deltas
        deltas = _compute_convergence_delta(self.model,
                                            self.layer,
                                            self.orig_dummy_input,
                                            input_dtypes,
                                            attributions,
                                            baselines,
                                            X,
                                            target,
                                            self._is_list,
                                            compute_layer_inputs_gradients)

        return attributions, deltas

    def _compute_attributions_tensor_input(self,
                                           X: Union[np.ndarray, tf.Tensor],
                                           baselines: Union[np.ndarray, tf.Tensor],
                                           target: Optional[List[int]],
                                           step_sizes: List[float],
                                           alphas: List[float],
                                           nb_samples: int,
                                           compute_layer_inputs_gradients: bool) -> Tuple:
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
            Interpolation parameter defining the points of the interal path.
        nb_samples
            Total number of samples.
        compute_layer_inputs_gradients
            In case of layers gradients, controls whether the gradients are computed for the layer's inputs or
            outputs. If True, gradients are computed for the layer's inputs, if False for the layer's outputs.

        Returns
        -------
            Tuple with integrated gradients attributions, deltas and predictions
        """
        if (len(self.model.output_shape) == 1
            or self.model.output_shape[-1] == 1) \
                and target is None:
            logger.warning("It looks like you are passing a model with a scalar output and target is set to `None`."
                           "If your model is a regression model this will produce correct attributions. If your model "
                           "is a classification model, targets for each datapoint must be defined. "
                           "Not defining the target may lead to incorrect values for the attributions."
                           "Targets can be either the true classes or the classes predicted by the model.")

        input_dtypes = [xx.dtype for xx in X]

        # define paths in features's or layers' space
        paths = np.concatenate([baselines + alphas[i] * (X - baselines) for i in range(self.n_steps)], axis=0)

        # define target paths
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
        else:
            target_paths = None

        if target_paths is not None:
            paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(self.internal_batch_size)
        else:
            paths_ds = tf.data.Dataset.from_tensor_slices(paths).batch(self.internal_batch_size)

        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # calculate gradients for batches
        batches = []
        for path in paths_ds:

            if target is not None:
                paths_b, target_b = path
            else:
                paths_b, target_b = path, None

            if self.layer is None:
                grads_b = _gradients_input(self.model, paths_b, target_b)

            else:
                grads_b = _gradients_layer(self.model,
                                           self.layer,
                                           self.orig_call,
                                           self.orig_dummy_input,
                                           paths_b,
                                           target_b,
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

        # calculate convergence deltas
        deltas = _compute_convergence_delta(self.model,
                                            self.layer,
                                            self.orig_dummy_input,
                                            input_dtypes,
                                            attributions,
                                            baselines,
                                            X,
                                            target,
                                            self._is_list,
                                            compute_layer_inputs_gradients)

        return attributions, deltas

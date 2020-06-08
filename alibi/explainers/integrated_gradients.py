import copy
import logging
import numpy as np
import string
import tensorflow as tf

from alibi.api.defaults import DEFAULT_DATA_INTGRAD, DEFAULT_META_INTGRAD
from alibi.utils.approximation_methods import approximation_parameters
from alibi.api.interfaces import Explainer, Explanation

from typing import Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _compute_convergence_delta(model: Union[tf.keras.models.Model, 'keras.models.Model'],
                               attributions: np.ndarray,
                               start_point: np.ndarray,
                               end_point: np.ndarray,
                               target: Union[None, np.ndarray, list]) -> np.ndarray:
    """
    Computes convergence deltas for each data point. Convergence delta measures how close the sum of all attributions
    is to the difference between the model output at the baseline and the model output at the data point.

    Parameters
    ----------
    model
        Tensorflow or keras model.
    attributions
        Attributions assigned by the integrated gradients method to each feature.
    start_point
        Baselines.
    end_point
        Data points.
    target
        Target for which the gradients are calculated for classification models.

    Returns
    -------
        Convergence deltas for each data point.
    """

    if end_point.shape[0] != attributions.shape[0]:
        raise ValueError("`attributions` and `end_point` must match on the first dimension "
                         "but found `attributions`: {} and `end_point`: {}".format(attributions.shape[0],
                                                                                   end_point.shape[0]))

    start_point = tf.convert_to_tensor(start_point, dtype=model.input.dtype)
    end_point = tf.convert_to_tensor(end_point, dtype=model.input.dtype)

    def _sum_rows(inp):

        input_str = string.ascii_lowercase[1: len(inp.shape)]
        if isinstance(inp, tf.Tensor):
            sums = tf.einsum('a{}->a'.format(input_str), inp).numpy()
        elif isinstance(inp, np.ndarray):
            sums = np.einsum('a{}->a'.format(input_str), inp)
        else:
            raise NotImplementedError('input must be a tensorflow tensor or a numpy array')
        return sums

    start_out = _run_forward(model, start_point, target)
    end_out = _run_forward(model, end_point, target)
    if (len(model.output_shape) == 1 or model.output_shape[1] == 1) and target is not None:
        target_tensor = tf.cast(target, dtype=start_out.dtype)
        sign = 2 * target_tensor - 1
        target_tensor = tf.reshape(1 - target_tensor, [len(target), 1])

        start_out = target_tensor + sign * start_out
        end_out = target_tensor + sign * end_out

    start_out_sum = _sum_rows(start_out)
    end_out_sum = _sum_rows(end_out)

    attr_sum = _sum_rows(attributions)

    _deltas = attr_sum - (end_out_sum - start_out_sum)

    return _deltas


def _run_forward(model: Union[tf.keras.models.Model, 'keras.models.Model'],
                 x: Union[tf.Tensor, np.ndarray],
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

    def _select_target(ps, ts):
        if ts is not None:
            if isinstance(ps, tf.Tensor):
                ps = tf.linalg.diag_part(tf.gather(ps, ts, axis=1))
            else:
                raise NotImplementedError
        else:
            raise ValueError("target cannot be `None` if `model` output dimensions > 1")
        return ps

    preds = model(x)
    if len(model.output_shape) > 1 and model.output_shape[1] > 1:
        preds = _select_target(preds, target)

    return preds


def _gradients_input(model: Union[tf.keras.models.Model, 'keras.models.Model'],
                     x: tf.Tensor,
                     target: Union[None, tf.Tensor]) -> tf.Tensor:
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


def _gradients_layer(model: Union[tf.keras.models.Model, 'keras.models.Model'],
                     layer: Union[tf.keras.layers.Layer, 'keras.layers.Layer'],
                     orig_call: Callable,
                     x: tf.Tensor,
                     target: Union[None, tf.Tensor]) -> tf.Tensor:
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
                # Store the result of `layer.call` internally.
                layer.result = func(*args, **kwargs)
                # From this point onwards, watch this tensor.
                tape.watch(layer.result)
                # Return the result to continue with the forward pass.
                return layer.result

            return wrapper

        layer.call = decorator(layer.call)
        return layer

    with tf.GradientTape() as tape:
        watch_layer(layer, tape)
        preds = _run_forward(model, x, target)

    grads = tape.gradient(preds, layer.result)

    delattr(layer, 'result')
    layer.call = orig_call

    return grads


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


def _format_input_baseline(X: np.ndarray,
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
        Formatted baselines as a numpy array.

    """
    if baselines is None:
        bls = np.zeros(X.shape).astype(X.dtype)
    elif isinstance(baselines, int) or isinstance(baselines, float):
        bls = np.full(X.shape, baselines).astype(X.dtype)
    elif isinstance(baselines, np.ndarray):
        bls = baselines.astype(X.dtype)
    else:
        raise ValueError('baselines must be `int`, `float`, `np.ndarray` or `None`. Found {}'.format(type(baselines)))

    return bls


def _format_target(target: Union[None, int, list, np.ndarray],
                   nb_samples: int) -> list:
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


class IntegratedGradients(Explainer):

    def __init__(self,
                 model: Union[tf.keras.Model, 'keras.Model'],
                 layer: Union[None, tf.keras.layers.Layer, 'keras.layers.Layer'] = None,
                 method: str = "gausslegendre",
                 n_steps: int = 50,
                 internal_batch_size: Union[None, int] = 100
                 ) -> None:
        """
        An mplementation of the integrated gradients method for Tensorflow and Keras models.

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
        if layer is None:
            layer_num = 0
        else:
            layer_num = model.layers.index(layer)
        params['layer'] = layer_num
        self.meta['params'].update(params)

        self.model = model
        self.input_dtype = self.model.input.dtype
        self.layer = layer
        self.n_steps = n_steps
        self.method = method
        self.internal_batch_size = internal_batch_size

    def explain(self,
                X: np.ndarray,
                baselines: Union[None, int, float, np.ndarray] = None,
                target: Union[None, int, list, np.ndarray] = None) -> Explanation:
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

        Returns
        -------
            `Explanation` object including `meta` and `data` attributes with integrated gradients attributions
            for each feature.

        """
        if not tf.executing_eagerly():
            raise RuntimeError("""To run IntegratedGradients Tensorflow must be executed eagerly.
            To enable eager execution, add the following lines at the beginning of your script:
            `import tensorflow as tf`
            `tf.compat.v1.enable_eager_execution()` """)

        if (len(self.model.output_shape) == 1 or self.model.output_shape[1] == 1) and target is None:
            logger.warning("It looks like you are passing a model with a scalar output and target is set to `None`."
                           "If your model is a regression model this will produce correct attributions. If your model "
                           "is a classification model, targets for each datapoint must be defined. "
                           "Not defining the target may lead to incorrect values for the attributions."
                           "Targets can be either the true classes or the classes predicted by the model.")

        nb_samples = len(X)

        # format and check inputs and targets
        baselines = _format_input_baseline(X, baselines)
        target = _format_target(target, nb_samples)

        # defining integral method
        step_sizes_func, alphas_func = approximation_parameters(self.method)
        step_sizes, alphas = step_sizes_func(self.n_steps), alphas_func(self.n_steps)

        # construct paths and prepare batches
        paths = np.concatenate([baselines + alphas[i] * (X - baselines) for i in range(self.n_steps)], axis=0)
        if target is not None:
            target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)
            paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(self.internal_batch_size)
        else:
            paths_ds = tf.data.Dataset.from_tensor_slices(paths).batch(self.internal_batch_size)
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        # fix orginal call method for layer
        if self.layer is not None:
            orig_call = self.layer.call
        else:
            orig_call = None

        # calculate gradients for batches
        batches = []
        for path in paths_ds:

            if target is not None:
                paths_b, target_b = path
            else:
                paths_b, target_b = path, None

            if self.layer is not None:
                grads_b = _gradients_layer(self.model, self.layer, orig_call,
                                           tf.dtypes.cast(paths_b, self.input_dtype), target_b)
            else:
                grads_b = _gradients_input(self.model,
                                           tf.dtypes.cast(paths_b, self.input_dtype), target_b)

            batches.append(grads_b)

        # tf concatatation
        grads = tf.concat(batches, 0)
        shape = grads.shape[1:]

        # invert sign of gradients for target 0 examples if classifier returns only positive class probability
        if (len(self.model.output_shape) == 1 or self.model.output_shape[1] == 1) and target is not None:
            sign = 2 * target_paths - 1
            grads = np.array([s * g for s, g in zip(sign, grads)])

        grads = tf.reshape(grads, (self.n_steps, nb_samples) + shape)

        # sum integral terms and scale attributions
        sum_int = _sum_integral_terms(step_sizes, grads.numpy())
        if self.layer is not None:
            norm = (self.layer(X) - self.layer(baselines)).numpy()
        else:
            norm = X - baselines
        attributions = norm * sum_int

        return self.build_explanation(
            X=X,
            baselines=baselines,
            target=target,
            attributions=attributions
        )

    def build_explanation(self,
                          X: np.ndarray,
                          baselines: np.ndarray,
                          target: list,
                          attributions: np.ndarray) -> Explanation:
        data = copy.deepcopy(DEFAULT_DATA_INTGRAD)
        data.update(X=X,
                    baselines=baselines,
                    target=target,
                    attributions=attributions)

        # calculate predictions
        predictions = self.model(X).numpy()
        data.update(predictions=predictions)

        # calculate convergence deltas
        deltas = _compute_convergence_delta(self.model, attributions, baselines, X, target)
        data.update(deltas=deltas)

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

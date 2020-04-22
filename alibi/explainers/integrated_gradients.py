import tensorflow as tf
import numpy as np
from typing import Callable, Tuple, Union, TYPE_CHECKING

from alibi.api.defaults import DEFAULT_META_INTGRAD, DEFAULT_DATA_INTGRAD
import logging
from alibi.utils.approximation_methods import approximation_parameters
from alibi.api.interfaces import Explainer, Explanation
import copy
import string

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

tf.compat.v1.enable_eager_execution()
logger = logging.getLogger(__name__)


def _compute_convergence_delta(forward_function: Union[tf.keras.models.Model, 'keras.models.Model'],
                               attributions: np.ndarray,
                               start_point: np.ndarray,
                               end_point: np.ndarray,
                               target: Union[np.ndarray, list]) -> np.ndarray:
    """

    Parameters
    ----------
    forward_function
    attributions
    start_point
    end_point
    target

    Returns
    -------

    """

    assert end_point.shape[0] == attributions.shape[0], (
        "Attributions and end_point must match on the first"
        " dimension but found attributions: {} and end_point: {}".format(
            attributions.shape[0], end_point.shape[0]))

    def _sum_rows(inp):
        if isinstance(inp, tf.Tensor):
            input_str = string.ascii_lowercase[1: len(inp.shape)]
            sums = tf.einsum('a{}->a'.format(input_str), inp).numpy()
        elif isinstance(inp, np.ndarray):
            input_str = string.ascii_lowercase[1: len(inp.shape)]
            sums = np.einsum('a{}->a'.format(input_str), inp)
        else:
            raise NotImplementedError('input must be a tf tensor or a np array')
        return sums

    start_out_sum = _sum_rows(_run_forward(forward_function, start_point, target))
    end_out_sum = _sum_rows(_run_forward(forward_function, end_point, target))

    attr_sum = _sum_rows(attributions)

    _deltas = attr_sum - (end_out_sum - start_out_sum)

    return _deltas


def _run_forward(forward_function: Union[tf.keras.models.Model, 'keras.models.Model'],
                 x: Union[tf.Tensor, np.ndarray],
                 target: Union[tf.Tensor, np.ndarray, list]) -> tf.Tensor:
    """

    Parameters
    ----------
    forward_function
    x
    target

    Returns
    -------

    """
    def _select_target(ps, ts):
        if ts is not None:
            if isinstance(ps, tf.Tensor):
                ps = tf.linalg.diag_part(tf.gather(ps, ts, axis=1))
            else:
                raise NotImplementedError
        else:
            raise ValueError("target cannot be None")
        return ps

    preds = forward_function(x)
    preds = _select_target(preds, target)

    return preds


def _gradients_input(forward_function: Union[tf.keras.models.Model, 'keras.models.Model'],
                     x: tf.Tensor,
                     target: tf.Tensor) -> tf.Tensor:
    """

    Parameters
    ----------
    forward_function
    x
    target

    Returns
    -------

    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = _run_forward(forward_function, x, target)

    grads = tape.gradient(preds, x)

    return grads


def _gradients_layer(forward_function: Union[tf.keras.models.Model, 'keras.models.Model'],
                     layer: Union[tf.keras.layers.Layer, 'keras.layers.Layer'],
                     orig_call: Callable,
                     x: tf.Tensor,
                     target: tf.Tensor) -> tf.Tensor:
    """

    Parameters
    ----------
    forward_function
    layer
    orig_call
    x
    target

    Returns
    -------

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
        preds = _run_forward(forward_function, x, target)

    grads = tape.gradient(preds, layer.result)

    delattr(layer, 'result')
    layer.call = orig_call

    return grads


def _sum_integral_terms(step_sizes: list,
                        grads: tf.Tensor) -> tf.Tensor:
    """

    Parameters
    ----------
    step_sizes
    grads

    Returns
    -------

    """
    step_sizes = tf.convert_to_tensor(step_sizes)
    input_str = string.ascii_lowercase[1: len(grads.shape)]
    einstr = 'a,a{}->{}'.format(input_str, input_str)

    return tf.einsum(einstr, step_sizes, grads)


def _sum_integral_terms_np(step_sizes: list,
                           grads: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    step_sizes
    grads

    Returns
    -------

    """
    input_str = string.ascii_lowercase[1: len(grads.shape)]
    einstr = 'a,a{}->{}'.format(input_str, input_str)

    return np.einsum(einstr, step_sizes, grads)


def _format_input_baseline(X: np.ndarray,
                           baselines: Union[None, int, float, np.ndarray]) -> Union[Tuple, np.ndarray]:
    """

    Parameters
    ----------
    X
    baselines

    Returns
    -------

    """
    if baselines is None:
        bls = np.zeros(X.shape)
    elif isinstance(baselines, int) or isinstance(baselines, float):
        bls = np.full(X.shape, baselines)
    else:
        raise ValueError('baselines must be int, float, np.ndarray or None. Found {}'.format(type(baselines)))

    assert len(X) == len(bls)

    return X, bls


def _format_target(target: Union[None, int, list, np.ndarray],
                   nb_samples: int) -> list:
    """

    Parameters
    ----------
    target
    nb_samples

    Returns
    -------

    """
    if target is not None:
        if isinstance(target, int):
            target = [target for _ in range(nb_samples)]
        elif isinstance(target, list) or isinstance(target, np.ndarray):
            pass
        else:
            raise NotImplementedError

    return target


class IntegratedGradients(Explainer):

    def __init__(self, forward_function: Union[tf.keras.Model, 'keras.Model'],
                 layer: Union[None, tf.keras.layers.Layer, 'keras.layers.Layer'] = None,
                 n_steps: int = 50,
                 method: str = "gausslegendre",
                 return_convergence_delta: bool = False,
                 return_predictions: bool = False):
        """

        Parameters
        ----------
        forward_function
        layer
        n_steps
        method
        return_convergence_delta
        return_predictions
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_INTGRAD))
        params = locals()
        remove = ['self', 'forward_function', '__class__', 'layer']
        params = {k: v for k, v in params.items() if k not in remove}
        self.meta['params'].update(params)

        self.forward_function = forward_function
        self.layer = layer
        self.n_steps = n_steps
        self.method = method
        self.return_convergence_delta = return_convergence_delta
        self.return_predictions = return_predictions

    def explain(self, X: np.ndarray,
                baselines: Union[None, int, float, np.ndarray] = None,
                features_names: Union[list, None] = None,
                target: Union[None, int, list, np.ndarray] = None,
                internal_batch_size: Union[None, int] = 100) -> Explanation:
        """

        Parameters
        ----------
        X
        baselines
        features_names
        target
        internal_batch_size

        Returns
        -------

        """
        nb_samples = len(X)

        X, baselines = _format_input_baseline(X, baselines)
        target = _format_target(target, nb_samples)
        assert len(target) == nb_samples

        step_sizes_func, alphas_func = approximation_parameters(self.method)
        step_sizes, alphas = step_sizes_func(self.n_steps), alphas_func(self.n_steps)

        paths = np.concatenate([baselines + alphas[i] * (X - baselines) for i in range(self.n_steps)], axis=0)
        target_paths = np.concatenate([target for _ in range(self.n_steps)], axis=0)

        paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(internal_batch_size)
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)

        batches = []
        if self.layer is not None:
            orig_call = self.layer.call
        for paths_b, target_b in paths_ds:

            # calculate gradients for batch
            if self.layer is not None:
                grads_b = _gradients_layer(self.forward_function, self.layer, orig_call, paths_b, target_b)
            else:
                grads_b = _gradients_input(self.forward_function, paths_b, target_b)

            batches.append(grads_b)

        # tf concatatation
        grads = tf.concat(batches, 0)
        shape = grads.shape[1:]
        grads = tf.reshape(grads, (self.n_steps, nb_samples) + shape)

        # sum integral terms
        sum_int = _sum_integral_terms_np(step_sizes, grads.numpy())
        if self.layer is not None:
            norm = (self.layer(X) - self.layer(baselines)).numpy()
        else:
            norm = X - baselines
        attr = norm * sum_int

        data = copy.deepcopy(DEFAULT_DATA_INTGRAD)
        data['X'] = X
        data['baselines'] = baselines
        data['attributions'] = attr
        data['features_names'] = features_names

        if self.return_predictions:
            predictions = self.forward_function(X).numpy()
            data['predictions'] = predictions

        if self.return_convergence_delta:
            deltas = _compute_convergence_delta(self.forward_function, attr, baselines, X, target)
            data['deltas'] = deltas

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

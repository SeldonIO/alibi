import numpy as np
from typing import Callable, Optional, Tuple, Union, TYPE_CHECKING
import tensorflow as tf

import logging
from alibi.utils.approximation_methods import approximation_parameters
from alibi.api.interfaces import Explainer, Explanation
import string
from time import time

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _compute_convergence_delta(forward_function: Callable,
                               attributions: np.ndarray,
                               start_point: np.ndarray,
                               end_point: np.ndarray,
                               target: Union[np.ndarray, list]) -> np.ndarray:

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


def _run_forward(forward_function: Callable,
                 x: Union[tf.Tensor, np.ndarray],
                 target: Union[tf.Tensor, np.ndarray, list]) -> tf.Tensor:

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


def _tf2_gradients(forward_function: Callable,
                   x: tf.Tensor,
                   target: tf.Tensor) -> tf.Tensor:

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = _run_forward(forward_function, x, target)

    grads = tape.gradient(preds, x)

    return grads


def _sum_integral_terms(step_sizes: int,
                        grads: tf.Tensor) -> tf.Tensor:

    step_sizes = tf.convert_to_tensor(step_sizes)
    input_str = string.ascii_lowercase[1: len(grads.shape)]
    einstr = 'a,a{}->{}'.format(input_str, input_str)

    return tf.einsum(einstr, step_sizes, grads)


def _format_input_baseline(X: np.ndarray,
                           baselines: Union[None, int, float, np.ndarray]) -> Union[Tuple, np.ndarray]:

    if baselines is None:
        bls = np.zeros(X.shape)
    elif isinstance(baselines, int) or isinstance(baselines, float):
        bls = np.full(X.shape, baselines)
    else:
        raise ValueError('baselines must be int, float, np.ndarray or None. Found {}'.format(type(baselines)))

    #if isinstance(X, np.ndarray):
        #X = tf.convert_to_tensor(X)

    assert len(X) == len(bls)

    return X, bls


def _format_target(target: Union[None, int, list, np.ndarray],
                   nb_samples: int) -> list:

    if target is not None:
        if isinstance(target, int):
            target = [target for _ in range(nb_samples)]
        elif isinstance(target, list) or isinstance(target, np.ndarray):
            pass
        else:
            raise NotImplementedError

    return target


class IntegratedGradientsTf(object):

    def __init__(self, forward_function: Callable, verbose=True):
        self.forward_function = forward_function
        self.verbose = verbose

    def explain(self, X: np.ndarray,
                baselines: Union[None, int, float, np.ndarray] = None,
                target: Union[None, int, list, np.ndarray] = None,
                n_steps: int = 50,
                method: string = "gausslegendre",
                internal_batch_size: Union[None, int] = None,
                return_convergence_delta: bool = False) -> Union[Tuple, np.ndarray]:

        t_0 = time()
        nb_samples = len(X)

        X, baselines = _format_input_baseline(X, baselines)
        target = _format_target(target, nb_samples)

        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        paths = np.concatenate([baselines + alphas[i] * (X - baselines) for i in range(n_steps)], axis=0)
        target_paths = np.concatenate([target for _ in range(n_steps)], axis=0)

        orig_shape = (n_steps,) + X.shape
        orig_shape_target = (n_steps, len(target))
        assert orig_shape[0] == orig_shape_target[0]
        t_paths = time() - t_0

        t_1 = time()
        paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(internal_batch_size)
        paths_ds.prefetch(tf.data.experimental.AUTOTUNE)
        t_batching = time() - t_1

        batches = []
        for paths_b, target_b in paths_ds:
            # calculate gradients for batch
            grads_b = _tf2_gradients(self.forward_function, paths_b, target_b)
            batches.append(grads_b)

        # tf concatatation
        grads = tf.concat(batches, 0)
        grads = tf.reshape(grads, orig_shape)

        # sum integral terms
        self.attr = (X - baselines) * _sum_integral_terms(step_sizes, grads).numpy()

        if return_convergence_delta:
            t_22 = time()
            deltas = self.compute_convergence_delta(baselines, X, target)
            t_delta = time() - t_22
            times = (t_paths, t_batching, t_delta)
            return self.attr, deltas, times
        else:
            times = (t_paths, t_batching)
            return self.attr, times

    def compute_convergence_delta(self, start_point, end_point, target):
        return _compute_convergence_delta(self.forward_function, self.attr, start_point, end_point, target)

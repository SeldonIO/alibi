import numpy as np
from typing import Callable, Optional, Tuple, Union, TYPE_CHECKING
import tensorflow as tf

import logging
from alibi.utils.approximation_methods import approximation_parameters
from time import time

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _compute_convergence_delta(forward_function,
                               attributions,
                               start_point,
                               end_point,
                               target):

    assert end_point.shape[0] == attributions.shape[0], (
        "Attributions and end_point must match on the first"
        " dimension but found attributions: {} and end_point: {}".format(
            attributions.shape[0], end_point.shape[0]))

    num_samples = end_point.shape[0]

    def _sum_rows(input):
        if isinstance(input, tf.Tensor):
            input = input.numpy()
        return input.reshape(input.shape[0], -1).sum(axis=1)

    start_out_sum = _sum_rows(_run_forward(forward_function, start_point, target))
    end_out_sum = _sum_rows(_run_forward(forward_function, end_point, target))

#    row_sums = [_sum_rows(attribution) for attribution in attributions]
#    print(len(row_sums), row_sums[0].shape)

#    attr_sum = torch.stack(
#    [cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)]
#    )

    attr_sum = _sum_rows(attributions)
    # attr_sum = attributions.sum(tuple(range(1, len(attributions.shape))))

    _delta = attr_sum - (end_out_sum - start_out_sum)

    return _delta


def _run_forward(forward_function, X, target):

    def _select_target(ps, ts):

        if isinstance(ps, tf.Tensor):
            if ts is not None:
                #ts = tf.convert_to_tensor(ts)
                ps = tf.linalg.diag_part(tf.gather(ps, ts, axis=1))
            else:
                raise AttributeError
        elif isinstance(ps, np.ndarray):
            raise NotImplementedError
        elif isinstance(ps, ''):
            raise NotImplementedError
        else:
            raise NotImplementedError

        return ps

    preds = forward_function(X)
    preds = _select_target(preds, target)

    return preds


def _tf2_gradients(forward_function, X, target):

    with tf.GradientTape() as tape:
        #X = tf.convert_to_tensor(X)
        tape.watch(X)
        preds = _run_forward(forward_function, X, target)

    grads = tape.gradient(preds, X)

    return grads


class IntegratedGradientsTf(object):

    def __init__(self, forward_function, verbose=True):
        self.forward_function = forward_function
        self.verbose = verbose

    def explain(self, X,
                baselines=None,
                target=None,
                additional_forward_args=None,
                n_steps=50,
                method="gausslegendre",
                internal_batch_size=None,
                return_convergence_delta=False):

        nb_samples = len(X)
        if baselines is None:
            baselines = np.zeros(X.shape)
        assert len(X) == len(baselines)

        if target is not None:
            if isinstance(target, int):
                target = [target for _ in range(nb_samples)]
            elif isinstance(target, list) or isinstance(target, np.ndarray):
                pass
            else:
                raise NotImplementedError

        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        paths = np.asarray([baselines + alphas[i] * (X - baselines) for i in range(n_steps)])
        target_paths = np.asarray([target for _ in range(n_steps)])
        orig_shape = paths.shape
        orig_shape_target = target_paths.shape
        assert orig_shape[0] == orig_shape_target[0]

        def _flatten(paths):
            return paths.reshape((paths.shape[0] * paths.shape[1],) + paths.shape[2:])
        paths = _flatten(paths)
        target_paths = _flatten(target_paths)

        paths_ds = tf.data.Dataset.from_tensor_slices((paths, target_paths)).batch(internal_batch_size)

        #nb_batches = np.ceil(len(paths) / internal_batch_size).astype(int)
        #print('Nb batches:', nb_batches)

        batches = []
        #for i in range(nb_batches):
        #    if self.verbose and i % 50 == 0:
        #        logger.info('processed {} batches of {}'.format(i, nb_batches))

        #    paths_b = paths[i * internal_batch_size:(i + 1) * internal_batch_size]
        #    target_b = target_paths[i * internal_batch_size:(i + 1) * internal_batch_size]
        for paths_b, target_b in paths_ds:
            # calculate gradients for batch
            grads_b = _tf2_gradients(self.forward_function, paths_b, target_b)
            #grads_b = grads_b.numpy()
            batches.append(grads_b)

        print('len of batches', len(batches))
        # numpy concatanation
        #grads = np.concatenate(batches, axis=0)
        # tf concatatation
        grads = tf.concat(batches, 0)
        print('grads shape', grads.shape)
        grads = grads.numpy().reshape(orig_shape)
        print('grads shape after reshaping', grads.shape)
        # sum integral terms
        self.attr = (X - baselines) * np.asarray([step_sizes[i] * grads[i] for i in range(n_steps)]).sum(axis=0)

        if return_convergence_delta:
            deltas = self.compute_convergence_delta(baselines, X, target)
            return self.attr, deltas
        else:
            return self.attr

    def compute_convergence_delta(self, start_point, end_point, target):
        return _compute_convergence_delta(self.forward_function, self.attr, start_point, end_point, target)
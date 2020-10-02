import inspect
import os
import pytest
import time

import numpy as np

from alibi.utils.distributed import DistributedExplainer, PoolCollection, ResourceError, invert_permutation, \
    concatenate_minibatches
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union


class MockExplainer:
    """ Mock explainer that allows testing most of the functionality in alibi.utils.distributed. """
    def __init__(self, sleep_time: int, multiplier: int = 3):
        self.sleep_time = sleep_time
        self.multiplier = multiplier
        self.proc_id = None  # type: Optional[int]

    def get_explanation(self, X: Union[Tuple[int, np.ndarray], np.ndarray], **kwargs):
        """
        Multiplies input by a constant and records process ID.
        """
        self.proc_id = os.getpid()
        time.sleep(self.sleep_time)
        if isinstance(X, tuple):
            batch_idx, X = X
            return batch_idx, self.multiplier*X
        else:
            return self.multiplier*X

    def return_attribute(self, name: str):
        """
        Returns an attribute specified by its name. Used in a distributed context where the actor properties cannot be
        accessed using the dot syntax.
        """
        return self.__getattribute__(name)


@pytest.fixture
def data_generator(request):
    """
    Generates a random integer matrix with values in `range(1, 10)`.
    """

    n_rows, n_columns = request.param
    return np.random.randint(1, 10, size=(n_rows, n_columns))


def kwargs_factory(keys: List[str], values: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Generates a list of dictionaries as follows:

        - each string in `keys` is a key
        - the elements of `values` are mapped through a cartesian product and then mapped to keys

    Parameters
    ----------
    keys:
        List of dict keys.
    values:
        List of dict values. See examples for result.

    Returns
    -------
        A list of dictionaries with keys specified in `keys`. Each element has as keys one of the elements of the
        cartesian product of the elements of `values`.

    Examples
    -------
        # >>> values, keys = [[1], [1, 2]], ['a', 'b']
        # >>> result = kwargs_factory(keys, values)
        # >>> print(result)
        [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    """

    n_values, n_keys = len(values), len(keys)
    if n_values != n_keys:
        raise ValueError(
            f"Missing value for a key. Specified {n_keys} keys, {n_values} values!"
        )

    return [dict(zip(keys, this_value)) for this_value in product(*values)]


def distributed_opts_id(params: dict):
    """
    Formatter to display distributed opts in test name.

    Params
    ------
    A dictionary containing `batch_size`, `n_cpus` and `actor_cpu_fraction` as keys.
    """
    fmt = 'batch_size={} n_cpus={} actor_cpu_fraction={}'
    return fmt.format(*list(params.values()))


def data_generator_id(params: List):
    """
    Formatter to display data dimensions in test name.

    Parameters
    ----------
    params
        Iterable with values for `n_instances` and `n_features`.
    """
    fmt = 'n_instances={}, n_features={}'
    return fmt.format(*params)


# generate distributed_opts dictionaries
batch_size = [1]
ncpus = [2]
actor_cpu_fraction = [0.5]
keys = ['batch_size', 'n_cpus', 'actor_cpu_fraction']
values = [batch_size, ncpus, actor_cpu_fraction]
distributed_opts = kwargs_factory(keys, values)  # type: ignore

# MockExplainer args and kwargs
explainer_init_args = [(0.01,), ]
explainer_init_kwargs = [{'multiplier': 2}, ]


@pytest.mark.parametrize('expln_args', explainer_init_args, ids='expln_init_args={}'.format)
@pytest.mark.parametrize('expln_kwargs', explainer_init_kwargs, ids='expln_init_kwargs={}'.format)
@pytest.mark.parametrize('distributed_opts', distributed_opts, ids=distributed_opts_id)
def test_distributed_explainer_init(expln_args, expln_kwargs, distributed_opts):

    import ray

    atol = 1e-5  # abs tolerance for floating point comparisons
    distributed_explainer = DistributedExplainer(distributed_opts, MockExplainer, expln_args, expln_kwargs)
    assert distributed_explainer.target_fcn.__name__ == 'default_target_fcn'
    assert distributed_explainer.concatenate
    assert ray.is_initialized()
    assert isinstance(distributed_explainer.pool, ray.util.ActorPool)
    assert len(distributed_explainer.pool._idle_actors) == distributed_opts['n_cpus']

    # test remote process is initialised properly
    for actor_idx in range(distributed_opts['n_cpus']):
        distributed_explainer.actor_index = actor_idx
        assert distributed_explainer.multiplier == expln_kwargs['multiplier']
        assert np.isclose(distributed_explainer.sleep_time, expln_args[0], atol=atol)

    ray.shutdown()


batch_size = [None, 1, 2]
ncpus = [2]
actor_cpu_fraction = [0.5]
keys = ['batch_size', 'n_cpus', 'actor_cpu_fraction']
values = [batch_size, ncpus, actor_cpu_fraction]
distributed_opts = kwargs_factory(keys, values)  # type: ignore


# MockExplainer args and kwargs
explainer_init_args = [(0.01,), ]
explainer_init_kwargs = [{'multiplier': 2}, ]

n_instances, n_features = 5, 6
return_generator = [False, True]  # whether a generator is returned or the results are computed and returned
concatenate_results = [False, True]


@pytest.mark.parametrize('data_generator', [(n_instances, n_features), ], ids=data_generator_id, indirect=True)
@pytest.mark.parametrize('expln_args', explainer_init_args, ids='expln_init_args={}'.format)
@pytest.mark.parametrize('expln_kwargs', explainer_init_kwargs, ids='expln_init_kwargs={}'.format)
@pytest.mark.parametrize('distributed_opts', distributed_opts, ids=distributed_opts_id)
@pytest.mark.parametrize('return_generator', return_generator, ids='return_generator={}'.format)
@pytest.mark.parametrize('concatenate_results', concatenate_results, ids='concat_results={}'.format)
def test_distributed_explainer_get_explanation(
        data_generator,
        expln_args,
        expln_kwargs,
        distributed_opts,
        return_generator,
        concatenate_results):

    import ray
    atol = 1e-5  # tolerance for numerical comparisons

    batch_size = distributed_opts['batch_size']
    ncpus = distributed_opts['n_cpus']
    distributed_explainer = DistributedExplainer(
        distributed_opts,
        MockExplainer,
        expln_args,
        expln_kwargs,
        return_generator=return_generator,
        concatenate_results=concatenate_results,
    )
    X = data_generator
    result = distributed_explainer.get_explanation(X)

    if return_generator:
        # test a generator is returned immediately
        assert inspect.isgenerator(result)
        # take all elems from generator so distributed execution can be tested
        for elem in result:
            assert isinstance(elem, tuple)
    else:
        # test batching
        if concatenate_results:
            assert isinstance(result, np.ndarray)
            assert result.shape == X.shape

        else:
            if batch_size is None:
                assert len(result) == ncpus  # split batch evenly if not specified
            elif X.shape[0] % batch_size == 0:
                assert len(result) == X.shape[0] // batch_size
            else:
                assert sum(res.shape[0] for res in result) == X.shape[0]
            # result correctness
            result = np.concatenate(result, axis=0).squeeze()
        assert np.isclose(np.unique(result / X), expln_kwargs['multiplier'], atol=atol)

    # test distributed execution
    proc_ids = []
    for idx in range(ncpus):
        distributed_explainer.actor_index = idx  # set this to retrieve state from different actors
        proc_ids.append(distributed_explainer.proc_id)
    assert len(set(proc_ids)) == ncpus

    # clean up after test
    ray.shutdown()


@pytest.fixture
def permutation_generator(request):
    """
    Generates a list containing a random permutation.
    """

    perm_len = request.param
    rng = np.random.default_rng()
    return rng.permutation(perm_len).tolist()


permutation_len = [6, ]


@pytest.mark.parametrize('permutation_generator', permutation_len, indirect=True)
def test_invert_permutation(permutation_generator):
    """
    Tests whether the inversion of the permutation is correctly calculated.
    """

    result = invert_permutation(permutation_generator)
    np.testing.assert_allclose(np.array(permutation_generator)[result], np.array(np.arange(result.shape[0])))


# MockExplainer args and kwargs
explainer_init_args = [[(0.01,), (0.05, )], ]   # type: ignore
explainer_init_kwargs = [[{'multiplier': 2}, {'multiplier': 3}], [{'multiplier': 2}, ]]  # type: ignore

batch_size = [2]
ncpus = [1, len(explainer_init_kwargs[0])]  # set so that both the error raising inputs and correct inputs are tested
actor_cpu_fraction = [1.0, 0.5, None]
keys = ['batch_size', 'n_cpus', 'actor_cpu_fraction']
values = [batch_size, ncpus, actor_cpu_fraction]
distributed_opts = kwargs_factory(keys, values)  # type: ignore
concatenate_results = [False, True]


@pytest.mark.parametrize('expln_args', explainer_init_args, ids='expln_init_args={}'.format)
@pytest.mark.parametrize('expln_kwargs', explainer_init_kwargs, ids='expln_init_kwargs={}'.format)
@pytest.mark.parametrize('distributed_opts', distributed_opts, ids=distributed_opts_id)
@pytest.mark.parametrize('concatenate_results', concatenate_results, ids='concat_results={}'.format)
def test_pool_collection_init(expln_args, expln_kwargs, distributed_opts, concatenate_results):

    import ray

    ncpus = distributed_opts['n_cpus']
    batch_size = distributed_opts['batch_size']
    cpu_frac = distributed_opts['actor_cpu_fraction']
    n_explainers = len(expln_args)

    if n_explainers != len(expln_kwargs):
        if ncpus > 1:
            with pytest.raises(ValueError):
                PoolCollection(
                    distributed_opts,
                    MockExplainer,
                    expln_args,
                    expln_kwargs,
                )
    elif ncpus == 1:
        if n_explainers == len(expln_kwargs):
            with pytest.raises(ResourceError):
                PoolCollection(
                    distributed_opts,
                    MockExplainer,
                    expln_args,
                    expln_kwargs,
                )
    else:
        explainer_collection = PoolCollection(
            distributed_opts,
            MockExplainer,
            expln_args,
            expln_kwargs,
            concatenate_results=concatenate_results,
        )

        assert ray.is_initialized()
        assert len(explainer_collection.distributed_explainers) == n_explainers
        for idx in range(n_explainers):
            explainer_collection.remote_explainer_index = idx
            assert explainer_collection.batch_size == batch_size
            assert explainer_collection.concatenate_results == concatenate_results
            if cpu_frac:
                assert explainer_collection.n_processes == (ncpus // n_explainers) // cpu_frac

    ray.shutdown()


# MockExplainer positional args and kwargs
explainer_init_args = [[(0.01,), (0.05, )], ]  # type: ignore
explainer_init_kwargs = [[{'multiplier': 2}, {'multiplier': 3}], ]  # type: ignore

batch_size = [2]
ncpus = [len(explainer_init_kwargs[0])]  # test only correct inputs
actor_cpu_fraction = [1.0]
keys = ['batch_size', 'n_cpus', 'actor_cpu_fraction']
values = [batch_size, ncpus, actor_cpu_fraction]
distributed_opts = kwargs_factory(keys, values)  # type: ignore
n_instances, n_features = 5, 6


@pytest.mark.parametrize('data_generator', [(n_instances, n_features), ], ids=data_generator_id, indirect=True)
@pytest.mark.parametrize('expln_args', explainer_init_args, ids='expln_init_args={}'.format)
@pytest.mark.parametrize('expln_kwargs', explainer_init_kwargs, ids='expln_init_kwargs={}'.format)
@pytest.mark.parametrize('distributed_opts', distributed_opts, ids=distributed_opts_id)
def test_pool_collection_get_explanation(data_generator, expln_args, expln_kwargs, distributed_opts):

    import ray
    atol = 1e-5  # absolute tolerance for floating point comparisons

    ncpus = distributed_opts['n_cpus']
    pool_collection = PoolCollection(
        distributed_opts,
        MockExplainer,
        expln_args,
        expln_kwargs,
        return_generator=False,  # just emphasizing this should always be the case, can't pickle generators
        concatenate_results=False,  # tested if concatenate_results works correctly already
    )

    X = data_generator

    explanations = pool_collection.get_explanation(X)
    for this_expln_result, this_expln_args, this_expln_kwargs in zip(explanations, expln_args, expln_kwargs):
        batched_result = np.concatenate(this_expln_result, axis=0).squeeze()
        assert np.isclose(np.unique(batched_result / X), this_expln_kwargs['multiplier'], atol=atol)

    proc_ids = []
    for idx in range(ncpus):
        pool_collection.remote_explainer_index = idx  # this is necessary to retrieve state from different processes
        for explainer_actor_idx in range(pool_collection.n_processes):
            pool_collection[idx].set_actor_index.remote(explainer_actor_idx)
            proc_ids.append(pool_collection.proc_id)
    assert len(set(proc_ids)) == ncpus

    ray.shutdown()


# repetitions: how many lists with n_minibatches of size minibatch_size are fed to the function in one go
n_minibatches, minibatch_size, repetitions, n_features = 3, 3, 3, 6
n_instances = n_minibatches*minibatch_size*repetitions


@pytest.mark.parametrize('data_generator', [(n_instances, n_features), ], ids=data_generator_id, indirect=True)
@pytest.mark.parametrize('n_minibatches', [n_minibatches, ], ids='n_minibatches={}'.format)
@pytest.mark.parametrize('repetitions', [repetitions, ], ids='repetitions={}'.format)
@pytest.mark.parametrize('n_features', [n_features, ], ids='n_features={}'.format)
def test_concatenate_minibatches(data_generator, n_minibatches, repetitions, n_features):
    """
    Tests the minibatch concatenation functions for all input types. Extend this test if you register new types.
    """

    X = data_generator

    # List[np.ndarray] input
    minibatches = np.split(X, n_minibatches)
    concat_result = concatenate_minibatches(minibatches)
    assert isinstance(concat_result, np.ndarray)
    np.testing.assert_allclose(X, concat_result)

    # List[List[np.ndarray]] input
    split_arr = np.split(X, n_minibatches*repetitions)
    minibatch_seq = [split_arr[i:i + n_minibatches] for i in range(0, len(split_arr), n_minibatches)]
    concat_result = concatenate_minibatches(minibatch_seq)

    # compute actual result
    slices = [(i, i + minibatch_size) for i in range(0, X.shape[0], minibatch_size)]
    grouped_slices = [slices[i::minibatch_size] for i in range(n_minibatches)]
    expected_result = [
        X[np.array([np.arange(*slc) for slc in grouped_slices[i]])].reshape(-1, n_features)
        for i in range(len(grouped_slices))
    ]

    assert isinstance(concat_result, list)
    assert len(concat_result) == len(minibatch_seq[0])
    for actual, expected in zip(expected_result, concat_result):
        np.testing.assert_allclose(actual, expected)

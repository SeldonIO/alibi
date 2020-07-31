import numpy as np

from functools import partial
from scipy import sparse
from typing import Any, Callable, Dict, List, Optional, Union


def check_ray():
    """
    Checks if ray is installed

    Returns:
    -------
        a bool indicating whether ray is installed or not
    """

    import importlib
    spec = importlib.util.find_spec('ray')
    if spec:
        return True
    return False


RAY_INSTALLED = check_ray()


class ActorPool(object):

    if RAY_INSTALLED:
        import ray
        ray = ray  # module as a static variable

    def __init__(self, actors):
        """
        Taken fom the ray repository: https://github.com/ray-project/ray/pull/5945
        Create an Actor pool from a list of existing actors.
        An actor pool is a utility class similar to multiprocessing.Pool that
        lets you schedule Ray tasks over a fixed pool of actors.
        Arguments:
            actors (list): List of Ray actor handles to use in this pool.
        Examples:
            # >>> a1, a2 = Actor.remote(), Actor.remote()
            # >>> pool = ActorPool([a1, a2])
            # >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
            [2, 4, 6, 8]
        """
        self._idle_actors = list(actors)
        self._future_to_actor = {}
        self._index_to_future = {}
        self._next_task_index = 0
        self._next_return_index = 0
        self._pending_submits = []

    def map(self, fn, values, chunksize=1):
        """Apply the given function in parallel over the actors and values.
        This returns an ordered iterator that will return results of the map
        as they finish. Note that you must iterate over the iterator to force
        the computation to finish.
        Arguments:
            fn (func): Function that takes (actor, value) as argument and
                returns an ObjectID computing the result over the value. The
                actor will be considered busy until the ObjectID completes.
            values (list): List of values that fn(actor, value) should be
                applied to.
            chunksize (int): splits the list of values to be submitted to the
                parallel process into sublists of size chunksize or less
        Returns:
            Iterator over results from applying fn to the actors and values.
        # Examples:
        #     >>> pool = ActorPool(...)
        #     >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
            [2, 4, 6, 8]
        """

        if chunksize:
            values = self._chunk(values, chunksize=chunksize)

        for v in values:
            self.submit(fn, v)
        while self.has_next():
            yield self.get_next()

    def map_unordered(self, fn, values, chunksize=1):
        """Similar to map(), but returning an unordered iterator.
        This returns an unordered iterator that will return results of the map
        as they finish. This can be more efficient that map() if some results
        take longer to compute than others.
        Arguments:
            fn (func): Function that takes (actor, value) as argument and
                returns an ObjectID computing the result over the value. The
                actor will be considered busy until the ObjectID completes.
            values (list): List of values that fn(actor, value) should be
                applied to.
            chunksize (int): splits the list of values to be submitted to the
                parallel process into sublists of size chunksize or less
        Returns:
            Iterator over results from applying fn to the actors and values.
        # Examples:
        #     >>> pool = ActorPool(...)
        #     >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
        #     [6, 2, 4, 8]
        """

        # if chunksize:
        #     values = self._chunk(values, chunksize=chunksize)

        for v in values:
            self.submit(fn, v)
        while self.has_next():
            yield self.get_next_unordered()

    def submit(self, fn, value):
        """Schedule a single task to run in the pool.
        This has the same argument semantics as map(), but takes on a single
        value instead of a list of values. The result can be retrieved using
        get_next() / get_next_unordered().
        Arguments:
            fn (func): Function that takes (actor, value) as argument and
                returns an ObjectID computing the result over the value. The
                actor will be considered busy until the ObjectID completes.
            value (object): Value to compute a result for.
        Examples:
            # >>> pool = ActorPool(...)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            # >>> print(pool.get_next(), pool.get_next())
            # 2, 4
        """
        if self._idle_actors:
            actor = self._idle_actors.pop()
            future = fn(actor, value)
            self._future_to_actor[future] = (self._next_task_index, actor)
            self._index_to_future[self._next_task_index] = future
            self._next_task_index += 1
        else:
            self._pending_submits.append((fn, value))

    def has_next(self):
        """Returns whether there are any pending results to return.
        Returns:
            True if there are any pending results not yet returned.
        Examples:
            # >>> pool = ActorPool(...)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            # >>> print(pool.has_next())
            # True
            # >>> print(pool.get_next())
            # 2
            # >>> print(pool.has_next())
            # False
        """
        return bool(self._future_to_actor)

    def get_next(self, timeout=None):
        """Returns the next pending result in order.
        This returns the next result produced by submit(), blocking for up to
        the specified timeout until it is available.
        Returns:
            The next result.
        Raises:
            TimeoutError if the timeout is reached.
        Examples:
            # >>> pool = ActorPool(...)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            # >>> print(pool.get_next())
            2
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        if self._next_return_index >= self._next_task_index:
            raise ValueError("It is not allowed to call get_next() after "
                             "get_next_unordered().")
        future = self._index_to_future[self._next_return_index]
        if timeout is not None:
            res, _ = self.ray.wait([future], timeout=timeout)
            if not res:
                raise TimeoutError("Timed out waiting for result")
        del self._index_to_future[self._next_return_index]
        self._next_return_index += 1
        i, a = self._future_to_actor.pop(future)
        self._return_actor(a)
        return self.ray.get(future)

    def get_next_unordered(self, timeout=None):
        """Returns any of the next pending results.
        This returns some result produced by submit(), blocking for up to
        the specified timeout until it is available. Unlike get_next(), the
        results are not always returned in same order as submitted, which can
        improve performance.
        Returns:
            The next result.
        Raises:
            TimeoutError if the timeout is reached.
        Examples:
            # >>> pool = ActorPool(...)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            # >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            # >>> print(pool.get_next_unordered())
            # 4
            # >>> print(pool.get_next_unordered())
            2
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        res, _ = self.ray.wait(
            list(self._future_to_actor), num_returns=1, timeout=timeout)
        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")
        i, a = self._future_to_actor.pop(future)
        self._return_actor(a)
        del self._index_to_future[i]
        self._next_return_index = max(self._next_return_index, i + 1)
        return self.ray.get(future)

    def _return_actor(self, actor):
        self._idle_actors.append(actor)
        if self._pending_submits:
            self.submit(*self._pending_submits.pop(0))

    @staticmethod
    def _chunk(values, chunksize):
        """Yield successive chunks of len=chunksize from values."""
        for i in range(0, len(values), chunksize):
            yield values[i:i + chunksize]


def kernel_shap_target_fn(actor: Any, instances: tuple, kwargs: Optional[Dict] = None) -> Callable:
    """
    A target function that is executed in parallel given an actor pool. Its arguments must be an actor and a batch of
    values to be processed by the actor. Its role is to execute distributed computations when an actor is available.

    Parameters
    ----------
    actor
        A `ray` actor. This is typically a class decorated with the @ray.remote decorator, that has been subsequently
        instantiated using cls.remote(*args, **kwargs).
    instances
        A (batch_index, batch) tuple containing the batch of instances to be explained along with a batch index.
    kwargs
        A list of keyword arguments for the actor `shap_values` method.

    Returns
    -------
    A callable that can be used as a target process for a parallel pool of actor objects.
    """

    if kwargs is None:
        kwargs = {}

    return actor.get_explanation.remote(instances, **kwargs)


def kernel_shap_postprocess_fn(ordered_result: List[Union[np.ndarray, List[np.ndarray]]]) \
        -> List[Union[np.ndarray, List[np.ndarray]]]:
    """
    Merges the results of the batched computation for KernelShap.

    Parameters
    ----------
    ordered_result
        A list containing the results for each batch, in the order that the batch was submitted to the parallel pool.
        It may contain:
            - `np.ndarray` objects (single-output predictor)
            - lists of `np.ndarray` objects (multi-output predictors)

    Returns
    -------
    concatenated
        A list containing the concatenated results for all the batches.
    """
    if isinstance(ordered_result[0], np.ndarray):
        return np.concatenate(ordered_result, axis=0)

    # concatenate explanations for every class
    n_classes = len(ordered_result[0])
    to_concatenate = [list(zip(*ordered_result))[idx] for idx in range(n_classes)]
    concatenated = [np.concatenate(arrays, axis=0) for arrays in to_concatenate]
    return concatenated


def invert_permutation(p: list):
    """
    Inverts a permutation.

    Parameters:
    -----------
    p
        Some permutation of 0, 1, ..., len(p)-1. Returns an array s, where s[i] gives the index of i in p.

    Returns
    -------
    s
        `s[i]` gives the index of `i` in `p`.
    """

    s = np.empty_like(p)
    s[p] = np.arange(len(p))
    return s


class DistributedExplainer:
    """
    A class that orchestrates the execution of the execution of a batch of explanations in parallel.
    """
    if RAY_INSTALLED:
        import ray
        ray = ray

    def __init__(self, distributed_opts, cls, init_args, init_kwargs):
        if not RAY_INSTALLED:
            raise ModuleNotFoundError("Module requires ray to be installed. pip install alibi[ray] ")

        self.n_jobs = distributed_opts['n_cpus']
        self.n_actors = int(distributed_opts['n_cpus'] // distributed_opts['actor_cpu_fraction'])
        self.actor_cpu_frac = distributed_opts['actor_cpu_fraction']
        self.batch_size = distributed_opts['batch_size']
        self.algorithm = distributed_opts['algorithm']
        self.target_fn = globals()[f"{distributed_opts['algorithm']}_target_fn"]
        try:
            self.post_process_fcn = globals()[f"{distributed_opts['algorithm']}_postprocess_fn"]
        except KeyError:
            self.post_process_fcn = None

        self.explainer = cls
        self.explainer_args = init_args
        self.explainer_kwargs = init_kwargs

        if not DistributedExplainer.ray.is_initialized():
            print(f"Initialising ray on {distributed_opts['n_cpus']} cpus!")
            DistributedExplainer.ray.init(num_cpus=distributed_opts['n_cpus'])

        self.pool = self.create_parallel_pool()

    def __getattr__(self, item):
        """
        Access to actor attributes. Should be used to retrieve only state that is shared by all actors in the pool.
        """
        actor = self.pool._idle_actors[0]
        return self.ray.get(actor.return_attribute.remote(item))

    def create_parallel_pool(self):
        """
        Creates a pool of actors (aka proceses containing explainers) that can execute explanations in parallel.
        """

        actor_handles = [
            DistributedExplainer.ray.remote(self.explainer).options(num_cpus=self.actor_cpu_frac)
            for _ in range(self.n_actors)
        ]

        actors = [handle.remote(*self.explainer_args, **self.explainer_kwargs) for handle in actor_handles]
        return DistributedExplainer.ray.util.ActorPool(actors)

    def batch(self, X: np.ndarray) -> enumerate:
        """
        Splits the input into sub-arrays according to the following logic:

            - if `batch_size` is not `None`, batches of this size are created. The sizes of the batches created might \
            vary if the 0-th dimension of `X` is not divisible by `batch_size`. For an array of length l that should be
            split into n sections, it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
            - if `batch_size` is `None`, then `X` is split into `n_jobs` sub-arrays

        Parameters
        ----------
        X
            Array to be split.
        Returns
        ------
            A list of sub-arrays of X.
        """

        n_records = X.shape[0]
        if isinstance(X, sparse.spmatrix):
            X = X.toarray()

        if self.batch_size:
            n_batches = n_records // self.batch_size
            if n_records % self.batch_size != 0:
                n_batches += 1
            slices = [self.batch_size*i for i in range(1, n_batches)]
            batches = np.array_split(X, slices)
        else:
            batches = np.array_split(X, self.n_jobs)
        return enumerate(batches)

    def get_explanation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Performs distributed explanations of instances in `X`.
        
        Parameters
        ----------
        X
            A batch of instances to be explained. Split into batches according to the settings passed to the constructor.
        kwargs
            Any keyword-arguments for the explainer `explain` method. 
            
        Returns
        --------
            An array of explanations.
        """ # noqa E501

        if kwargs is not None:
            self.target_fn = partial(self.target_fn, kwargs=kwargs)
        batched_instances = self.batch(X)

        unordered_explanations = self.pool.map_unordered(self.target_fn, batched_instances)

        return self.order_result(unordered_explanations)

    def order_result(self, unordered_result: List[tuple]) -> np.ndarray:
        """
        Re-orders the result of a distributed explainer so that the explanations follow the same order as the input to
        the explainer.


        Parameters
        ----------
        unordered_result
            Each tuple contains the batch id as the first entry and the explanations for that batch as the second.

        Returns
        -------
        A numpy array where the the batches ordered according to their batch id are concatenated in a single array.
        """

        # TODO: THIS DOES NOT LEVERAGE THE FACT THAT THE RESULTS ARE RETURNED AS AVAILABLE. ISSUE TO BE RAISED.

        result_order, results = list(zip(*[(idx, res) for idx, res in unordered_result]))
        orig_order = invert_permutation(list(result_order))
        ordered_result = [results[idx] for idx in orig_order]
        if self.post_process_fcn is not None:
            return self.post_process_fcn(ordered_result)
        return ordered_result

import copy
import logging

import numpy as np

from functools import partial
from scipy import sparse
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def check_ray() -> bool:
    """
    Checks if ray is installed.

    Returns:
    -------
        A bool indicating whether ray is installed or not.
    """

    import importlib
    spec = importlib.util.find_spec('ray')
    if spec:
        return True
    return False


RAY_INSTALLED = check_ray()


class ActorPool(object):

    # TODO: JANIS: IF YOU DECIDE TO TAKE A DEPENDENCY ON RAY CORE, REMOVE THIS AND IMPORT ActorPool FROM ray.util

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
            >>> a1, a2 = Actor.remote(), Actor.remote()
            >>> pool = ActorPool([a1, a2])
            >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
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
        Examples:
            >>> pool = ActorPool(...)
            >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
            [2, 4, 6, 8]
        """

        if chunksize > 1:
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
        Examples:
            >>> pool = ActorPool(...)
            >>> print(pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
            [6, 2, 4, 8]
        """

        if chunksize:
            values = self._chunk(values, chunksize=chunksize)

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
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            >>> print(pool.get_next(), pool.get_next())
            2, 4
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
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> print(pool.has_next())
            True
            >>> print(pool.get_next())
            2
            >>> print(pool.has_next())
            False
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
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> print(pool.get_next())
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
            >>> pool = ActorPool(...)
            >>> pool.submit(lambda a, v: a.double.remote(v), 1)
            >>> pool.submit(lambda a, v: a.double.remote(v), 2)
            >>> print(pool.get_next_unordered())
            4
            >>> print(pool.get_next_unordered())
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


def batch(X: np.ndarray, batch_size: Optional[int] = None, n_batches: int = 4) -> List[np.ndarray]:
    """
    Splits the input into sub-arrays.

    Parameters
    ----------
    X
        Array to be split.
    batch_size
        The size of each batch. In particular:

            - if `batch_size` is not `None`, batches of this size are created. The sizes of the batches created
            might vary if the 0-th dimension of `X` is not divisible by `batch_size`. For an array of length `l`
            that should be split into `n` sections, it returns `l % n` sub-arrays of size `l//n + 1` and the rest of
            `size l//n`
            - if `batch_size` is `None`, then `X` is split into `n_jobs` sub-arrays.
    n_batches
        Number of batches in which to split the sub-array. Only used if `batch_size = None`

    Returns
    ------
        A list of sub-arrays of X.
    """

    n_records = X.shape[0]
    if isinstance(X, sparse.spmatrix):
        logger.warning("Batching function received sparse matrix input. Converting to dense matrix first...")
        X = X.toarray()

    if batch_size:
        n_batches = n_records // batch_size
        if n_records % batch_size != 0:
            n_batches += 1
        slices = [batch_size * i for i in range(1, n_batches)]
        batches = np.array_split(X, slices)
    else:
        batches = np.array_split(X, n_batches)

    return batches


def default_target_fcn(actor: Any, instances: tuple, kwargs: Optional[Dict] = None):
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
        A list of keyword arguments for the actor `get_explanation` method.

    Returns
    -------
    A future that can be used to later retrieve the results of a distributed computation.

    Notes
    -----
    This function can be customized (e.g., if one does not desire to wrap the explainer such that it has
    `get_explanation` method. The customized function should be called `*_target_fcn` with the wildcard being replaced
    by the name of the explanation method (e.g., cem, cfproto, etc). The same name should be added to the
    `distributed_opts` dictionary passed by the user prior to instantiating the `DistributedExplainer`.

    """

    if kwargs is None:
        kwargs = {}

    return actor.get_explanation.remote(instances, **kwargs)


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


class ResourceError(Exception):
    pass


class DistributedExplainer:
    """
    A class that orchestrates the execution of the execution of a batch of explanations in parallel.
    """
    if RAY_INSTALLED:
        import ray
        ray = ray

    def __init__(self,
                 distributed_opts: Dict[str, Any],
                 explainer_type: Any,
                 explainer_init_args: Tuple,
                 explainer_init_kwargs: dict,
                 keep_order: bool = True):
        """
        Creates a pool of actors (i.e., replicas of an instantiated `explainer_type` in a separate process) which can 
        explain batches of instances in parallel via calls to `get_explanation`. 
        
        
        Parameters
        ----------
        distributed_opts
            A dictionary with the following type minimal signature::

                class DistributedOpts(TypedDict):
                    n_cpus: Optional[int]
                    batch_size: Optional[int]
            The dictionary may contain two additional keys:
                
                - ``'actor_cpu_frac'`` (float, <= 1.0, >0.0): This is used to create more than one process on one \\
                CPU/GPU. This may not speed up CPU intensive tasks but it is worth experimenting with when few physical \\ 
                cores are available. In particular, this is highly useful when the user wants to share a GPU for \\
                multiple tasks. See the ``ray`` documentation `here_` for details.
                
                .. _here:
                   https://docs.ray.io/en/stable/resources.html#fractional-resources
                
                - ``'algorithm'``: this is specified internally by the caller. It is used in order to register callbacks \\
                for results postprocessing or target functions for the parallel pool. These should be implemented in the \\
                global scope. If not specified, its value will be ``'default'``, meaning that the results will not be \\
                post-processed and that the default target function will be used. 
        explainer_type
            Explainer class.
        explainer_init_args
            Positional argument to explainer constructor.
        explainer_init_kwargs
            Keyword arguments to explainer constructor.
        keep_order
            If `True`, the raw values (e.g., attributions, the counterfactual instance values) of the explanations 
            returned follow the same same order as the instances, after the application of a custom post-processing 
            function if the later is specified. Otherwise, a generator that returns the explanations as they are 
            computed is returned. This is useful for returning results as they complete for heavy workloads, rather 
            than waiting until a large number of instances are explained.
        """ # noqa W605

        # TODO: TBD: @JANIS - DO WE HAVE TO DO ANY SETUP FOR THIS COMMAND TO WORK?
        if not RAY_INSTALLED:
            raise ModuleNotFoundError("Module requires ray to be installed. pip install alibi[ray] ")

        self.n_processes = distributed_opts['n_cpus']
        self.batch_size = distributed_opts['batch_size']
        self.keep_order = keep_order
        algorithm = distributed_opts.get('algorithm', 'default')
        if 'algorithm' == 'default':
            logger.warning(
                "No algorithm specified in distributed option, default target function and postprocessing will be "
                "selected."
            )
        # TODO: TBD: @JANIS - THE POSTPROCESSING COULD BE HANDLED BY CALLER BUT THAT COULD BE A BIT MESSIER?
        self.target_fcn = default_target_fcn
        self.post_process_fcn = None
        # check global scope for any specific target or result postprocessing function
        if f"{algorithm}_target_fn" in globals():
            self.target_fcn = globals()[f"{algorithm}_target_fn"]
        if f"{algorithm}_postprocess_fn" in globals():
            self.post_process_fcn = globals()[f"{algorithm}_postprocess_fn"]

        if not DistributedExplainer.ray.is_initialized():
            DistributedExplainer.ray.init(num_cpus=distributed_opts['n_cpus'])

        # a pool is a collection of handles to different processes that can process data points in parallel
        self.pool = self.create_parallel_pool(
            explainer_type,
            explainer_init_args,
            explainer_init_kwargs
        )

    def __getattr__(self, item: str, actor_index: int = 0) -> Any:
        """
        Accesses actor attributes. Use sparingly as this involves a remote call (that is, these attributes are of an
        object in a different process). The intended use is for retrieving any common state across the actor at the end
        of the computation in order to form the response.

        Parameters
        ----------
        actor_index
            The actor from which the state is to be retrieved.
        item
            The explainer attribute to be returned.

        Returns
        -------
            The value of the attribute specified by `item`.

        Raises
        ------
        ValueError
            If the actor index is invalid.

        Notes
        -----
        This method assumes that the actor implements a `return_attribute` method.
        """

        if actor_index > self.n_processes - 1:
            raise ValueError(f"Index of actor should be less than or equal to {self.n_processes - 1}!")

        actor = self.pool._idle_actors[actor_index]  # noqa
        return actor.return_attribute.remote(item)

    def create_parallel_pool(self, explainer_type: Any, explainer_init_args: Tuple, explainer_init_kwargs: dict):
        """
        Creates a pool of actors that can explain the rows of a dataset in parallel.

        Parameters
        ----------
        See constructor documentation.
        """

        handles = [DistributedExplainer.ray.remote(explainer_type) for _ in range(self.n_processes)]
        workers = [handle.remote(*explainer_init_args, **explainer_init_kwargs) for handle in handles]
        return DistributedExplainer.ray.util.ActorPool(workers)

    def get_explanation(self, X: np.ndarray, **kwargs) -> \
            Union[np.ndarray, List[np.ndarray], Iterator[Union[np.ndarray, List[np.ndarray]]]]:
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
        """  # noqa E501

        if kwargs is not None:
            self.target_fcn = partial(self.target_fcn, kwargs=kwargs)
        batched_instances = batch(X, self.batch_size, self.n_processes)
        unordered_explanations = self.pool.map_unordered(self.target_fcn, batched_instances)

        if self.keep_order:
            return self.order_result(unordered_explanations)
        return unordered_explanations

    def order_result(self, unordered_result: List[tuple]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Re-orders the result of a distributed explainer so that the explanations follow the same order as the input to
        the explainer. If no algorithm specific post-processing function is implemented in the global scope, then the
        result is a list

        Parameters
        ----------
        unordered_result
            Each tuple contains the batch id as the first entry and the explanations for that batch as the second.

        Returns
        -------
        A numpy array where the the batches ordered according to their batch id are concatenated in a single array.

        """

        # TODO: THIS DOES NOT LEVERAGE THE FACT THAT THE RESULTS ARE RETURNED AS AVAILABLE. THIS CAN BE IMPROVED.
        #  THE FACT THAT RAY GENERATES THE UNORDERED RESULT ONE BY ONE (AND DOES NOT WAIT FOR ALL TASKS TO FINISH)
        #  CAN BE LEVERAGED TO POPULATE UI GRADUALLY. HERE I JUST WAIT FOR ALL RESULTS TO COMPLETE AND THEN PERMUTE THEM
        #  AND RETURN ... TO ALLOW THIS TO WORK BETTER IN PROD, I ALLOW TO RETURN THE

        result_order, results = list(zip(*[(idx, res) for idx, res in unordered_result]))
        orig_order = invert_permutation(list(result_order))
        ordered_result = [results[idx] for idx in orig_order]
        if self.post_process_fcn is not None:
            return self.post_process_fcn(ordered_result)
        return ordered_result


class PoolCollection:
    """
    A wrapper object that turns a DistributedExplainer into a remote actor. This allows running multiple distributed
    explainers in parallel.
    """

    if RAY_INSTALLED:
        import ray
        ray = ray

    def __init__(self,
                 distributed_opts: Dict[str, Any],
                 explainer_type: Any,
                 explainer_init_args: List[Tuple],
                 explainer_init_kwargs: List[Dict]):
        """
        Initialises a list of *distinct* distributed explainers which can explain the same batch in parallel. It
        generalizes the `DistributedExplainer`, which contains replicas of one explainer object, speeding up the task
        of explaining batches of instances.

        Parameters
        ----------
        distributed_opts , explainer_type, explainer_init_args, explainer_init_kwargs
            See DistributedExplainer constructor documentation for explanations. Each entry in the list is a
            different explainer configuration (e.g., CEM in PN vs PP mode, different background dataset sizes for SHAP,
            etc).

        Raises
        ------
        ResourceError
            If the number of CPUs specified by the user is smaller than the number of distributed explainers.
        ValueError
            If the number of entries in the explainers args/kwargs list differ.
        """


        available_cpus = distributed_opts['n_cpus']
        if len(explainer_init_args) != len(explainer_init_kwargs):
            raise ValueError(
                "To run multiple explainers over distinct parallel pools of replicas, the lists of args and kwargs"
                "should be of equal length."
            )
        cpus_per_pool = available_cpus // len(explainer_init_args)
        if cpus_per_pool < 1:
            raise ResourceError(
                f"Running {explainer_type.__name__} requires {len(explainer_init_args)} CPU but only {available_cpus} "
                f"were specified. Please allocate more cpus to run this explainer in distributed mode"
            )
        # we can allow users to experiment with CPU fractions if only 1 CPU available per pool
        actor_cpu_fraction = distributed_opts.get('actor_cpu_fraction', 1.0)
        if cpus_per_pool == 1:
            if actor_cpu_fraction is not None:
                cpus_per_pool /= actor_cpu_fraction

        if not PoolCollection.ray.is_initialized():
            PoolCollection.ray.init(num_cpus=distributed_opts['n_cpus'])

        opts = copy.deepcopy(distributed_opts)
        opts.update(n_cpus=cpus_per_pool)
        self.distributed_explainers = self.create_explainer_handles(
            opts,
            explainer_type,
            explainer_init_args,
            explainer_init_kwargs,
        )

    @staticmethod
    def create_explainer_handles(distributed_opts: Dict[str, Any],
                                 explainer_type: Any,
                                 explainer_init_args: List[Tuple],
                                 explainer_init_kwargs: List[Dict]):
        """
        Creates multiple actors for DistributedExplainer so that tasks can be executed in parallel. The actors are
        initialised with different arguments, so they represent different explainers.
        """

        explainer_handles = [PoolCollection.ray.remote(DistributedExplainer) for _ in range(len(explainer_init_args))]
        distributed_explainers = []
        for handle, exp_args, exp_kwargs in zip(explainer_handles, explainer_init_args, explainer_init_kwargs):
            distributed_explainers.append(handle.remote(distributed_opts, explainer_type, exp_args, exp_kwargs))

        return distributed_explainers

    def get_explanation(self, X, **kwargs) -> List:
        """
        Calls a collection of distributed explainers in parallel. Each distributed explainer will explain each row in
        `X` in parallel.

        Parameters
        ----------
        X
            Batch of instances to be explained.

        Returns
        -------
        A list of responses collected from each explainer.

        Notes
        -----
        Note that the call to `ray.get` is blocking.

        """

        # TODO: Janis: I think that this can be improved for the case when the DistributedExample object returns
        #  immediately (this might be as simple as returning the futures instead of calling ray.get)
        return PoolCollection.ray.get(
            [explainer.get_explanation.remote(X, **kwargs) for explainer in self.distributed_explainers]
        )

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

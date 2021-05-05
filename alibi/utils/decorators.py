from functools import singledispatch, update_wrapper


def methdispatch(func):
    """
    A decorator that is used to support singledispatch style functionality for instance methods. By default,
    `functools.singledispatch` selects a function to call from registered based on the type of args[0]::

        def wrapper(*args, **kw):
            return dispatch(args[0].__class__)(*args, **kw)

    This uses `functools.singledispatch` to do achieve this but instead uses ``args[1]`` since ``args[0]`` will always
    be an instance of the object.
    """

    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)

    return wrapper

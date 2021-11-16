"""
This module defines the Alibi exception hierarchy and common exceptions
used across the library.
"""
from abc import ABC


class AlibiException(Exception, ABC):
    """
    Abstract base class of all alibi exceptions.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AlibiPredictorCallException(AlibiException):
    """
    This exception is raised whenever a call to a user supplied predictor fails at runtime.
    """
    pass


class AlibiPredictorReturnTypeError(AlibiException):
    """
    This exception is raised whenever the return type of a user supplied predictor is of
    an unexpected or unsupported type.
    """
    pass

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


# Legacy exception classes starting with the prefix `Alibi`.
# These have been kept around for backwards compatibility.
# Any new exception classes should not start with the `Alibi` prefix.
class AlibiPredictorCallException:
    pass


class AlibiPredictorReturnTypeError:
    pass


# Exception classes. These should only inherit from `AlibiException`. Ones inheriting from
# other classes beginning with the prefix `Alibi` are to do with backwards compatibility as
# exception classes used to all start with the prefix `Alibi`.`
class PredictorCallError(AlibiException, AlibiPredictorCallException):
    """
    This exception is raised whenever a call to a user supplied predictor fails at runtime.
    """
    pass


class PredictorReturnTypeError(AlibiException, AlibiPredictorReturnTypeError):
    """
    This exception is raised whenever the return type of a user supplied predictor is of
    an unexpected or unsupported type.
    """
    pass

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


class NotFittedError(AlibiException):
    """
    This exception is raised whenever a compulsory call to a `fit` method has not been carried out.
    """

    def __init__(self, object_name: str):
        super().__init__(
            f"This {object_name} instance is not fitted yet. Call 'fit' with appropriate arguments first."
        )


class SerializationError(AlibiException):
    """
    This exception is raised whenever an explainer cannot be serialized.
    """
    def __init__(self, message: str):
        super().__init__(message)

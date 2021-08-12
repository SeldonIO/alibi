"""
This module contains a loss wrapper and a definition of various monitoring metrics used during training. The model
to be trained inherits form :py:class:`alibi.explainers.models.pytorch.model.Model` and represents a simplified
version of the `tensorflow.keras` API for training and monitoring the model. Currently it is used internally to test
the functionalities for the Pytorch backend. To be discussed if the module will be exposed to the user in future
versions.
"""

import torch
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Union, Callable


class Reduction(Enum):
    """ Reduction operation supported by the monitoring metrics. """
    SUM = 'sum'
    MEAN = 'mean'


class LossContainer:
    """ Loss wrapped to monitor the average loss throughout training. """

    def __init__(self, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], name: str):
        """
        Constructor.

        Parameters
        ----------
        loss
            Loss function.
        name
            Name of the loss function
        """
        self.name = name
        self.loss = loss
        self.total, self.count = 0., 0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes and accumulates the loss given the prediction labels and the true labels.

        Parameters
        ----------
        y_pred
            Prediction labels.
        y_true
            True labels.
        """

        # compute loss
        loss = self.loss(y_pred, y_true)

        # add loss to the total
        self.total += loss.item()
        self.count += 1
        return loss

    def result(self) -> Dict[str, float]:
        """
        Computes the average loss obtain by dividing the cumulated loss by the number of steps

        Returns
        -------
        Average loss.
        """
        return {self.name: self.total / self.count}

    def reset(self):
        """ Resets the loss. """
        self.total = 0
        self.count = 0


class Metric(ABC):
    """
    Monitoring metric object. Supports two types of reduction: mean and sum.
    """

    def __init__(self, reduction: Reduction = Reduction.MEAN, name: str = "unknown"):
        """
        Constructor.

        Parameters
        ----------
        reduction
            Metric's reduction type. Possible values `mean`|`sum`. By default `mean`.
        name
            Name of the metric.
        """
        self.reduction = reduction
        self.name = name
        self.total, self.count = 0, 0

    @abstractmethod
    def compute_metric(self,
                       y_pred: Union[torch.Tensor, np.ndarray],
                       y_true: Union[torch.Tensor, np.ndarray]):
        pass

    def update_state(self, values: np.ndarray):
        """
        Update the state of the metric by summing up the metric values and updating the counts by adding
        the number of instances for which the metric was computed (first dimension).
        """
        self.total += values.sum()
        self.count += values.shape[0]

    def result(self) -> Dict[str, float]:
        """
        Computes the result according the the reduction procedure.

        Returns
        -------
        Monitoring metric.
        """
        if self.reduction == Reduction.SUM:
            return {self.name: self.total}

        if self.reduction == Reduction.MEAN:
            return {self.name: np.nan_to_num(self.total / self.count)}

        raise NotImplementedError(f"Reduction {self.reduction} not implemented")

    def reset(self):
        """ Resets the monitoring metric. """
        self.total = 0
        self.count = 0


class AccuracyMetric(Metric):
    """ Accuracy monitoring metric. """

    def __init__(self, name: str = "accuracy"):
        super().__init__(reduction=Reduction.MEAN, name=name)

    def compute_metric(self,
                       y_pred: Union[torch.Tensor, np.ndarray],
                       y_true: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Computes accuracy metric given the predicted label and the true label.

        Parameters
        ----------
        y_pred
            Predicted label.
        y_true
            True label.
        """

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # in case y_pred is a distribution and not a label
        if len(y_pred.shape) > len(y_true.shape):
            y_pred = np.argmax(y_pred, axis=-1)

        # check if the shapes match
        if y_pred.shape != y_true.shape:
            raise ValueError("The shape of the prediction and labels do not match")

        matches = np.array(y_pred == y_true)
        super().update_state(matches)

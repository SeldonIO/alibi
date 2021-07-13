import torch
import numpy as np
from enum import Enum
from typing import  Dict, Union


class Reduction(Enum):
    SUM = 'sum'
    MEAN = 'mean'


class LossContainer:
    def __init__(self, loss):
        self.name = "loss"
        self.loss = loss
        self.total, self.count = 0, 0

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # compute loss
        loss = self.loss(y_pred, y_true)

        # add loss to the total
        self.total += loss.item()
        self.count += 1
        return loss

    def result(self) -> Dict[str, float]:
        return {self.name: self.total / self.count}

    def reset(self):
        self.total = 0
        self.count = 0


class Metric:
    def __init__(self, reduction: Reduction = Reduction.MEAN, name: str = "unknown"):
        self.reduction = reduction
        self.name = name
        self.total, self.count = 0, 0

    def update_state(self, values: np.ndarray):
        self.total += values.sum()
        self.count += values.shape[0]

    def result(self) -> Dict[str, float]:
        if self.reduction == Reduction.SUM:
            return {self.name: self.total}

        if self.reduction == Reduction.MEAN:
            return {self.name: np.nan_to_num(self.total / self.count)}

        raise NotImplementedError(f"Reduction {self.reduction} not implemented")

    def reset(self):
        self.total = 0
        self.count = 0


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__(reduction=Reduction.MEAN, name="accuracy")

    def update_state(self, y_true: Union[torch.Tensor, np.ndarray], y_pred: [torch.Tensor, np.ndarray]):
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

        matches = (y_pred == y_true)
        super().update_state(matches)

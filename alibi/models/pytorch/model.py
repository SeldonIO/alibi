import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import List, Dict, Callable, Union, Tuple, Any, Optional
from alibi.models.pytorch.metrics import Metric, LossContainer


class NamedModule(nn.Module):
    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, *input):
        return self.module(*input)


class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(self,
                optimizer: optim.Optimizer,
                loss: Union[Callable, List[Callable]],
                loss_weights: Optional[Union[float, List[float]]] = None,
                metrics: Optional[List[Metric]] = None):
        """
        Compiles a model by setting the optimizer and the loss.

        Parameters
        ----------
        optimizer
            Optimizer to be used.
        loss
            Loss function to be used.
        """
        self.optimizer = optimizer
        self.metrics = [] if metrics is None else metrics
        self.loss_weights = [] if loss_weights is None else loss_weights

        if isinstance(loss, list):
            # check if the number of weights is the same as the number of partial losses
            if len(loss_weights) != len(loss):
                raise ValueError("The number of loss weights differs from the number of losses")

            self.loss = []
            for i, partial_loss in enumerate(loss):
                self.loss.append(LossContainer(partial_loss, name=f"output_{i+1}_loss"))
        else:
            self.loss = LossContainer(loss, name="loss")

    def validate_prediction_labels(self,
                                   y_pred: Union[torch.Tensor, List[torch.Tensor]],
                                   y_true: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(self.loss, list):
            # check that prediction is a list
            if not isinstance(y_pred, list):
                raise ValueError("The prediction should be a list since list of losses have been passed.")

            # check that the labels is a list
            if not isinstance(y_true, list):
                raise ValueError("The label should be a list since list of losses have been passed.")

            # check if the number of predictions matches the number of labels
            if len(y_true) != len(y_pred):
                raise ValueError("Number of predictions differs from the number of labels.")

            # check if the number of output heads matches the number of output losses
            if len(y_pred) != len(self.loss):
                raise ValueError("Number of model's heads differs from the number of losses.")
        else:
            # check that the prediction is not a list
            if isinstance(y_pred, list):
                raise ValueError("The prediction is a list and should be a tensor since only one loss has been passed")

            # check that the label is not a list
            if isinstance(y_true, list):
                raise ValueError("The label is a list and should be a tensor since only one loss has been passed")

        # check if metrics and predictions agree
        if (len(self.metrics) > 0) and (not isinstance(self.metrics, dict)) and isinstance(y_pred, list):
            raise ValueError("Multiple model's head require dictionary of metrics.")

    def compute_loss(self,
                     y_pred: Union[torch.Tensor, List[torch.Tensor]],
                     y_true: Union[torch.Tensor, List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # compute loss
        if isinstance(self.loss, list):
            loss = 0
            results = dict()

            for i, partial_loss in enumerate(self.loss):
                weight = self.loss_weights[i] if len(self.loss_weights) else 1.
                loss += weight * partial_loss(y_pred[i], y_true[i])
                results.update({key: weight * val for key, val in partial_loss.result().items()})

            # compute total loss
            results.update({"loss": sum(results.values())})
        else:
            loss = self.loss(y_pred, y_true)
            results = self.loss.result()
        return loss, results

    def compute_metrics(self,
                        y_pred: Union[torch.Tensor, List[torch.Tensor]],
                        y_true: Union[torch.Tensor, List[torch.Tensor]]):
        results = dict()

        if isinstance(self.metrics, dict):
            for name in self.metrics:
                i = int(name.split("_")[1]) - 1  # name is of the form output_1_... . Maybe we use re?
                self.metrics[name].update_state(y_pred=y_pred[i], y_true=y_true[i])

                # add output prefix in front of the results
                result = {name + "_" + key: val for key, val in self.metrics[name].result().items()}
                results.update(result)

        else:  # this is just for one head
            for metric in self.metrics:
                metric.update_state(y_pred=y_pred, y_true=y_true)
                results.update(metric.result())

        return results

    def train_step(self, x: torch.Tensor, y: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, float]:
        """
        Performs a train step.

        Parameters
        ----------
        x
            Input tensor.
        y
            Label tensor.
        """
        # set model to train
        self.train()

        # send tensors to device
        x = x.to(self.device)
        y_true = [y_i.to(self.device) for y_i in y] if isinstance(y, list) else y.to(self.device)

        # compute output
        y_pred = self.forward(x)

        # validate prediction and labels
        self.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

        # compute loss
        loss, results = self.compute_loss(y_pred=y_pred, y_true=y_true)

        # perform gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update metrics
        metrics = self.compute_metrics(y_pred=y_pred, y_true=y_true)
        results.update(metrics)
        return results

    @torch.no_grad()
    def test_step(self,
                  x: torch.Tensor,
                  y: Union[torch.Tensor, List[torch.tensor]]):
        """
        Performs a test step.

        Parameters
        ----------
        x
            Input tensor.
        y
            Label tensor.
        """
        # set to evaluation
        self.eval()

        # sent tensors to device
        x = x.to(self.device)
        y_true = [y_i.to(self.device) for y_i in y] if isinstance(y, list) else y.to(self.device)

        # compute output
        y_pred = self.forward(x)

        # validate prediction and labels
        self.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

        # compute loss
        loss, results = self.compute_loss(y_pred=y_pred, y_true=y_true)

        # update metrics
        metrics = self.compute_metrics(y_pred=y_pred, y_true=y_true)
        results.update(metrics)
        return results

    def fit(self, trainloader: DataLoader, epochs: int) -> Dict[str, float]:
        for epoch in range(epochs):
            print("Epoch %d/%d" % (epoch, epochs))

            # reset losses and metrics
            self._reset_loss()
            self._reset_metrics()

            # perform train steps in batches
            for data in tqdm(trainloader):
                if len(data) < 2:
                    raise ValueError("An input and at least a label should be provided")

                x = data[0]
                y = data[1] if len(data) == 2 else data[1:]
                metrics_vals = self.train_step(x, y)

            # print metrics
            print(Model._metrics_to_str(metrics_vals))
        return metrics_vals

    def evaluate(self, testloader: DataLoader) -> Dict[str, float]:
        self._reset_loss()
        self._reset_metrics()

        # perform test steps in batches
        for data in tqdm(testloader):
            if len(data) < 2:
                raise ValueError("An input and at least a label should be provided.")

            x = data[0]
            y = data[1] if len(data) == 2 else data[1:]
            metrics_vals = self.test_step(x, y)

        # log losses
        print(Model._metrics_to_str(metrics_vals))
        return metrics_vals

    @staticmethod
    def _metrics_to_str(metrics: Dict[str, float]) -> str:
        str_losses = ''
        for key in metrics:
            str_losses += "%s: %.4f\t" % (key, metrics[key])
        return str_losses

    def _reset_loss(self):
        if isinstance(self.loss, list):
            for partial_loss in self.loss:
                partial_loss.reset()
        else:
            self.loss.reset()

    def _reset_metrics(self):
        metrics = self.metrics.values() if isinstance(self.metrics, dict) else self.metrics
        for metric in metrics:
            metric.reset()

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
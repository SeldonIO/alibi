import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import List, Dict, Callable
from alibi.models.pytorch.metrics import Metric, LossContainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(self, optimizer: optim.Optimizer, loss: Callable, metrics: List[Metric] = []):
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
        self.loss = LossContainer(loss)
        self.metrics = metrics

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
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
        y = y.to(self.device)

        # compute loss
        out = self.forward(x)
        loss = self.loss(out, y)

        # perform gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # dictionary of results
        results = self.loss.result()

        # update metrics
        for metric in self.metrics:
            metric.update_state(y_true=y, y_pred=out)
            results.update(metric.result())

        return results

    @torch.no_grad()
    def test_step(self, x: torch.Tensor, y: torch.Tensor):
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
        y = y.to(self.device)

        # compute output
        out = self.forward(x)

        # compute loss
        self.loss(out, y)
        results = self.loss.result()

        # update metrics
        for metric in self.metrics:
            metric.update_state(y_true=y, y_pred=out)
            results.update(metric.result())

        return results

    def fit(self, trainloader: DataLoader, epochs: int) -> Dict[str, float]:
        for epoch in range(epochs):
            print("Epoch %d/%d" % (epoch, epochs))

            # reset losses and metrics
            self.loss.reset()
            for metric in self.metrics:
                metric.reset()

            # perform train steps in batches
            for x, y in tqdm(trainloader):
                metrics_vals = self.train_step(x, y)

            # print metrics
            print(Model._metrics_to_str(metrics_vals))
        return metrics_vals

    def evaluate(self, testloader: DataLoader) -> Dict[str, float]:
        self.loss.reset()
        for metric in self.metrics:
            metric.reset()

        # perform test steps in batches
        for x, y in tqdm(testloader):
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

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
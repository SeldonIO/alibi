import torch
import torch.nn as nn
from alibi.models.pytorch.model import Model


class Classifier(Model):
    def __init__(self, classifier_net: nn.Module, name: str = 'classifier'):
        super().__init__()
        self.classifier_net = classifier_net
        self.name = name

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_net(x)
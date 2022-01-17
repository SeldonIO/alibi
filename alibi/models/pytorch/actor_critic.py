"""
This module contains the Pytorch implementation of actor-critic networks used in the Counterfactual with Reinforcement
Learning for both data modalities. The models' architectures follow the standard actor-critic design and can have
broader use-cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
    The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
    tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        output_dim
            Output dimension
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.LazyLinear(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.LazyLinear(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Continuous action.
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """
    Critic network. The network follows the standard actor-critic architecture used in Deep Reinforcement Learning.
    The model is used in Counterfactual with Reinforcement Learning (CFRL) for both data modalities (images and
    tabular). The hidden dimension used for the all experiments is 256, which is a common choice in most benchmarks.
    """

    def __init__(self, hidden_dim: int):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.LazyLinear(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Critic value.
        """
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

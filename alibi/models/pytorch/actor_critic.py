import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """ Actor network """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Constructor.

        Parameters
        ----------
        input_dim
            Input dimension.
        hidden_dim
            Hidden dimension.
        output_dim
            Output dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """ Critic network """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Constructor.

        Parameters
        ----------
        input_dim
            Input dimension.
        hidden_dim
            Hidden dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x




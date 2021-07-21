import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from alibi.models.pytorch.model import Model


class MNISTClassifier(Model):
    """ Mnist classifier. """

    def __init__(self, output_dim: int) -> None:
        """
        Constructor.

        Parameters
        ----------
        output_dim
            Output dimension.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 2), padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1568, 256)
        self.fc2 = nn.Linear(256, output_dim)

        # send to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.maxpool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.maxpool2(F.relu(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTEncoder(nn.Module):
    """ MNIST Encoder. """

    def __init__(self, latent_dim: int):
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=2)
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        return x


class MNISTDecoder(nn.Module):
    """ MNIST Decoder. """

    def __init__(self, latent_dim: int):
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 128)
        self.conv1 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3, 3))
        self.up3 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = x.view(x.shape[0], 8, 4, 4)
        x = self.up1(F.relu(self.conv1(x)))
        x = self.up2(F.relu(self.conv2(x)))
        x = self.up3(F.relu(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x


class ADULTEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        """
        Constructor.

        Parameters
        ----------
        input_dim
            Input dimension.
        hidden_dim
            Hidden dimension.
        latent_dim
            Latent dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class ADULTDecoder(nn.Module):
    """ Adult decoder. """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dims: List[int]):
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        hidden_dim
            Hidden dimension.
        output_dims
            List of output dimensions.
        """
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in output_dims])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.fc1(x)
        xs = self.fcs(x)
        return xs

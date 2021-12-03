"""
This module contains the Pytorch implementation of models used for the Counterfactual with Reinforcement Learning
experiments for both data modalities (image and tabular).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from alibi.models.pytorch.model import Model


class MNISTClassifier(Model):
    """
    MNIST classifier used in the experiments for Counterfactual with Reinforcement Learning. The model consists of two
    convolutional layers having 64 and 32 channels and a kernel size of 2 with ReLU nonlinearities, followed by
    maxpooling of size 2 and dropout of 0.3. The convolutional block is followed by a fully connected layer of 256 with
    ReLU nonlinearity, and finally a fully connected layer is used to predict the class logits (10 in MNIST case).
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Classification logits.
        """
        x = self.dropout1(self.maxpool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.maxpool2(F.relu(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTEncoder(nn.Module):
    """
    MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
    consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
    Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
    follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
    in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
    this can vary.
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Encoding representation having each component in the interval [-1, 1]
        """
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.tanh(self.fc1(x))
        return x


class MNISTDecoder(nn.Module):
    """
    MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
    connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
    consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
    except the last one, has ReLU nonlinearities and is followed by an upsampling layer of size 2. The final layers
    uses a sigmoid activation to clip the output values in [0, 1].
    """

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
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Decoded input having each component in the interval [0, 1].
        """
        x = F.relu(self.fc1(x))
        x = x.view(x.shape[0], 8, 4, 4)
        x = self.up1(F.relu(self.conv1(x)))
        x = self.up2(F.relu(self.conv2(x)))
        x = self.up3(F.relu(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        return x


class ADULTEncoder(nn.Module):
    """
    ADULT encoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    two fully connected layers with ReLU and tanh nonlinearities. The tanh nonlinearity clips the embedding in [-1, 1]
    as required in the DDPG algorithm (e.g., [act_low, act_high]). The layers' dimensions used in the paper are
    128 and 15, although those can vary as they were selected to generalize across many datasets.
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        latent_dim
            Latent dimension.
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
       Forward pass.

       Parameters
       ----------
       x
           Input tensor.

       Returns
       -------
       Encoding representation having each component in the interval [-1, 1]
       """
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class ADULTDecoder(nn.Module):
    """
    ADULT decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    of a fully connected layer with ReLU nonlinearity, and a multiheaded layer, one for each categorical feature and
    a single head for the rest of numerical features. The hidden dimension used in the paper is 128.
    """

    def __init__(self, hidden_dim: int, output_dims: List[int]):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        output_dims
            List of output dimensions.
        """
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fcs = nn.ModuleList([nn.LazyLinear(dim) for dim in output_dims])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        List of reconstruction of the input tensor. First element corresponds to the reconstruction of all the \
        numerical features if they exist, and the rest of the elements correspond to each categorical feature.
        """
        x = F.relu(self.fc1(x))
        xs = [fc(x) for fc in self.fcs]
        return xs

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights, resnet50


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, classes: int) -> None:
        self.model = resnet50(pretrained=True, weights=ResNet50_Weights.IMAGENET1K_V2)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))

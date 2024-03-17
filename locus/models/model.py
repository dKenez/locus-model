from typing import Literal, Union

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


class LDoGIResnet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, classes: int, layers: Union[Literal[18], Literal[34], Literal[50], Literal[101], Literal[152]]):
        super(LDoGIResnet, self).__init__()

        # match case of layer to resnet model
        match layers:
            case 18:
                self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            case 34:
                self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
            case 50:
                self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            case 101:
                self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
            case 152:
                self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)

        # Modify the fully connected layer for the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.resnet(x)
        # Apply softmax to get probabilities
        return torch.softmax(x, dim=1)

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class LDoGIResnet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, classes: int) -> None:
        super(LDoGIResnet, self).__init__()

        # Load pretrained ResNet50 model
        self.resnet = resnet50(pretrained=True, weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modify the fully connected layer for the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, classes)

    def forward(self, x):
        # Forward pass through the network
        return self.resnet(x)

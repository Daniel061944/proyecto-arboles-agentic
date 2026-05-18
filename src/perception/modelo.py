import torch
import torch.nn as nn
from torchvision import models


class TreeResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def crear_modelo(num_classes: int, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = TreeResNet18(num_classes).to(device)
    return modelo
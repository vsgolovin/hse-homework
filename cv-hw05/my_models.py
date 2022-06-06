from typing import Iterable
import torch
from torch import nn
from torchvision import models


class WhoNeedsConvnets(nn.Module):
    def __init__(self, hidden_dims: Iterable[int],
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        stack = [nn.Flatten()]
        inp_dim = 3 * 32 * 32
        for out_dim in hidden_dims:
            stack.append(nn.Linear(inp_dim, out_dim))
            stack.append(activation())
            inp_dim = out_dim
        stack.append(nn.Linear(inp_dim, 10))
        self.stack = nn.Sequential(*stack)

    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.stack(X)


class MyVGG(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_features: bool = False,
                 bn: bool = True):
        super().__init__()
        if not bn:
            self.features = models.vgg11(pretrained=pretrained).features
        else:
            self.features = models.vgg11(pretrained=pretrained).features
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 10)
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.features(X)
        X = self.flatten(X)
        return self.classifier(X)

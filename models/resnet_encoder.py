import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        # remove classification head
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return features
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import resnet18, ResNet18_Weights


class ResNetAudio(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetAudio, self).__init__()
        # Load a pre-trained ResNet with updated weight loading method
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


def initialize_model(num_classes=6):
    model = ResNetAudio(num_classes=num_classes)
    return model

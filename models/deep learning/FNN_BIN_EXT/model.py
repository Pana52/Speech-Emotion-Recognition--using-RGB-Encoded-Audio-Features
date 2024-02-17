import torch.nn as nn
import torch.nn.functional as F


class AudioClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

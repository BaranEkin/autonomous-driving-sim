import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock


class ResFirst2(ResNet):
    def __init__(self):
        super(ResFirst2, self).__init__(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetAutoSteer(nn.Module):
    def __init__(self):
        super(ResNetAutoSteer, self).__init__()

        self.backbone = ResFirst2()
        self.layers = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x / 127.5 - 1.0
        x = self.backbone(x)
        x = self.layers(x)
        return x

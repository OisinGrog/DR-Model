import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class pretrained_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.resnet50.children())[:-1], nn.Flatten(), nn.Linear(2048, 2048))
        self.fc = nn.Linear(2048, 2)

    def forward(self, input):
        x = self.model(input)
        x = F.relu(x)
        x = self.fc(x)

        return x

# data_1 = torch.randn(16, 3, 224, 224)
# model_1 = pretrained_Resnet()
# output = model_1(data_1)

import torch
import torch.nn as nn
import torch.nn.functional as F
# dhdhdh
class Example3DCNN(nn.Module):
    def __init__(self):
        super(Example3DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected/Dense Layer
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, input):
        """

        :param input: Static Image of size 3 x 224 x 224
        :return:
        """
        print(f'Input shape: {input.shape}')
        # B = input.shape[0]
        x = self.pool(F.relu(self.bn1(self.conv1(input))))
        print(f'Output of X after CNN 1: {x.shape}')
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        print(f'Output of X after CNN 2: {x.shape}')
        x = F.relu(self.bn3(self.conv3(x)))
        print(f'Output of X after CNN 3: {x.shape}')
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        print(f'Output of X after CNN 4: {x.shape}')
        x = F.relu(self.bn5(self.conv5(x)))
        print(f'Output of X after CNN 5: {x.shape}')
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        print(f'Output of X after CNN 6: {x.shape}')
        # print("Size before flattening:", x.size())

        # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        print(f'Output of X after Flatenning: {x.shape}')
        x = self.fc2(x)
        print(f'Shape of output class: {x.shape}')
        return x


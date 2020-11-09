import torch.nn as nn
import torch.nn.functional as F


class Convnet(nn.Module):

    def __init__(self):
        super(Convnet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=(2, 2))
        self.fc1 = nn.Linear(256*2*2, 256)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))
        x = self.dropout(self.pool(F.relu(self.conv5(x))))
        x = x.view(-1, 256*2*2)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
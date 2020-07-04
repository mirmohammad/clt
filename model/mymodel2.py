import torch
import torch.nn as nn

from torchvision import models


class MyResnet1(nn.Module):

    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=6)

        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        return x


class MyNet1(nn.Module):

    def __init__(self):
        super(MyNet1, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(11, 3), stride=1, padding=(5, 1), dilation=1, bias=False),
            nn.BatchNorm2d(16),
        )
        # 320 x 180
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 11), stride=1, padding=(1, 5), dilation=1, bias=False),
            nn.BatchNorm2d(16),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 160 x 90
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 160 x 90
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 80 x 45
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 80 x 45
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1)

        # 20 x 15
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 20 x 15
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 20 x 15
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1)

        # 5 x 5
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 5 x 5
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )

        # 5 x 5
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = torch.cat((self.conv1_h(x), self.conv1_w(x)), dim=1)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))

        return x


class MyNet2(nn.Module):

    def __init__(self):
        super(MyNet2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(11, 3), stride=1, padding=(5, 1), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 320 x 180
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 11), stride=1, padding=(1, 5), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 160 x 90
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 160 x 90
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        # 80 x 45
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 80 x 45
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1)

        # 20 x 15
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 20 x 15
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 20 x 15
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1)

        # 5 x 5
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )
        # 5 x 5
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )

        # 5 x 5
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = torch.cat((self.conv1_h(x), self.conv1_w(x)), dim=1)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))

        return x
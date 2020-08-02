from abc import ABC

import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 5), stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 316 x 178
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 158 x 89
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 5), stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 154 x 87
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 5), stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 150 x 85
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3), stride=2, padding=0, dilation=1)
        # 74 x 42
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 70 x 40
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 5), stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 66 x 38
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 33 x 19
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 5), stride=2, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 15 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 5), stride=2, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 6 x 4
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel2(nn.Module):

    def __init__(self):
        super(MyModel2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=(2, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 106 x 60
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 53 x 30
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 53 x 30
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 53 x 30
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3), stride=2, padding=0, dilation=1)
        # 26 x 15
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 26 x 15
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 26 x 15
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1)
        # 13 x 7
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 7 x 4
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 7 x 4
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel3(nn.Module):

    def __init__(self):
        super(MyModel3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=(2, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 106 x 60
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 53 x 30
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 53 x 30
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 3), stride=2, padding=0, dilation=1)
        # 26 x 15
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 26 x 15
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1)
        # 13 x 7
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 7 x 4
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel4(nn.Module):

    def __init__(self):
        super(MyModel4, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 3), stride=(5, 1), padding=(3, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 11), stride=(1, 5), padding=(1, 3), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool1_w(self.relu(self.conv1_h(x))) + self.pool1_h(self.relu(self.conv1_w(x)))
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel5(nn.Module):

    def __init__(self):
        super(MyModel5, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(11, 3), stride=(5, 1), padding=(3, 1), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 11), stride=(1, 5), padding=(1, 3), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = torch.cat((self.pool1_w(self.relu(self.conv1_h(x))), self.pool1_h(self.relu(self.conv1_w(x)))), dim=1)

        x = self.conv2_1(x)
        id1 = x
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MyModel6(nn.Module):

    def __init__(self):
        super(MyModel6, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(11, 3), stride=(5, 1), padding=(3, 1), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 11), stride=(1, 5), padding=(1, 3), dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool1_w(self.relu(self.conv1_h(x))) + self.pool1_h(self.relu(self.conv1_w(x)))
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel7(nn.Module):

    def __init__(self):
        super(MyModel7, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(21, 3), stride=(3, 1), padding=(0, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 320 x 54
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6), padding=(0, 2), dilation=1)
        # 54 x 54
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 54 x 54
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 54 x 54
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 27 x 27
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 27 x 27
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 27 x 27
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1)
        # 13 x 13
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 13 x 13
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 13 x 13
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel8(nn.Module):

    def __init__(self):
        super(MyModel8, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(15, 3), stride=(5, 1), padding=(5, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 15), stride=(1, 5), padding=(1, 5), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = torch.cat((self.pool1_w(self.relu(self.conv1_h(x))), self.pool1_h(self.relu(self.conv1_w(x)))), dim=1)
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel9(nn.Module):

    def __init__(self):
        super(MyModel9, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(15, 3), stride=(5, 1), padding=(5, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 15), stride=(1, 5), padding=(1, 5), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool1_w(self.relu(self.conv1_h(x))) + self.pool1_h(self.relu(self.conv1_w(x)))
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel10(nn.Module):

    def __init__(self):
        super(MyModel10, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(16),
        )
        # 320 x 180
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(16),
        )
        # 320 x 180
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(16),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 160 x 90
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 160 x 90
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
        )
        # 160 x 90
        self.conv2_3 = nn.Sequential(
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
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2), padding=0, dilation=1)
        # 40 x 15
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 40 x 15
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 40 x 15
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 40 x 15
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2), padding=0, dilation=1)
        # 20 x 5
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.pool5 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc = nn.Linear(256, 6)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        id1 = x
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        id4 = x
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x += id4
        x = self.relu(x)
        x = self.pool5(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyModel11(nn.Module):

    def __init__(self):
        super(MyModel11, self).__init__()

        blocks = [2, 2, 3, 3, 3]
        kernel = [3, 64, 128, 256, 512, 512]

        for i in range(len(blocks)):
            block = []
            for j in range(blocks[i]):
                z = i if j == 0 else i + 1
                block.append(nn.Conv2d(kernel[z], kernel[i + 1], kernel_size=3, padding=1))
                block.append(nn.BatchNorm2d(kernel[i + 1]))
                block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            setattr(self, 'encode' + str(i + 1), nn.Sequential(*block))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MySegNet1(nn.Module):

    def __init__(self, num_classes=21):
        super(MySegNet1, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        kernel = [64, 128, 256, 512, 512]

        # 320 x 180
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.conv12 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 160 x 90

        # 160 x 90
        self.conv21 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.conv22 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 80 x 45

        # 80 x 45
        self.conv31 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv33 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1, return_indices=True)
        # 40 x 22

        # 40 x 22
        self.conv41 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.conv42 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.conv43 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 20 x 11

        # 20 x 11
        self.conv51 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.conv52 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.conv53 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1, return_indices=True)
        # 10 x 5

        kernel = [512, 512, 256, 128, 64]

        # 10 x 5
        self.unpool5 = nn.MaxUnpool2d(kernel_size=(3, 2), stride=2, padding=0)
        # 20 x 11
        self.deconv53 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 20 x 11
        self.deconv52 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 20 x 11
        self.deconv51 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 20 x 11

        # 20 x 11
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 40 x 22
        self.deconv43 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 40 x 22
        self.deconv42 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 40 x 22
        self.deconv41 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 40 x 22

        # 40 x 22
        self.unpool3 = nn.MaxUnpool2d(kernel_size=(3, 2), stride=2, padding=0)
        # 80 x 45
        self.deconv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.deconv31 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 80 x 45

        # 80 x 45
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 160 x 90
        self.deconv22 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 160 x 90
        self.deconv21 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 160 x 90

        # 160 x 90
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 320 x 180
        self.deconv12 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 320 x 180
        self.deconv11 = nn.Sequential(
            nn.Conv2d(kernel[4], 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(2),
        )
        # 320 x 180

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(kernel[0], 6)

    def forward(self, x):

        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x, idx1 = self.pool1(x)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.relu(x)
        x, idx2 = self.pool2(x)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.relu(x)
        x, idx3 = self.pool3(x)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x = self.relu(x)
        x, idx4 = self.pool4(x)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x = self.relu(x)
        x = self.conv53(x)
        x = self.relu(x)
        x, idx5 = self.pool5(x)

        y = self.avgpool(x)
        y = self.fc(y.squeeze())

        x = self.unpool5(x, indices=idx5)
        x = self.deconv53(x)
        x = self.relu(x)
        x = self.deconv52(x)
        x = self.relu(x)
        x = self.deconv51(x)
        x = self.relu(x)

        x = self.unpool4(x, indices=idx4)
        x = self.deconv43(x)
        x = self.relu(x)
        x = self.deconv42(x)
        x = self.relu(x)
        x = self.deconv41(x)
        x = self.relu(x)

        x = self.unpool3(x, indices=idx3)
        x = self.deconv32(x)
        x = self.relu(x)
        x = self.deconv31(x)
        x = self.relu(x)

        x = self.unpool2(x, indices=idx2)
        x = self.deconv22(x)
        x = self.relu(x)
        x = self.deconv21(x)
        x = self.relu(x)

        x = self.unpool1(x, indices=idx1)
        x = self.deconv12(x)
        x = self.relu(x)
        x = self.deconv11(x)
        x = self.relu(x)

        return y, x


# 3.08 (Epoch 82)
class MySegNet2(nn.Module):

    def __init__(self, num_classes=21):
        super(MySegNet2, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        kernel = [16, 32, 64, 128, 128]

        # 320 x 180
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.conv12 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 160 x 90

        # 160 x 90
        self.conv21 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.conv22 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 80 x 45

        # 80 x 45
        self.conv31 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv33 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1, return_indices=True)
        # 40 x 22

        # 40 x 22
        self.conv41 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.conv42 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.conv43 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 40 x 22
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 20 x 11

        # 20 x 11
        self.conv51 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.conv52 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.conv53 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 20 x 11
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 2), stride=2, padding=0, dilation=1, return_indices=True)
        # 10 x 5

        kernel = [128, 128, 64, 32, 16]

        # 10 x 5
        self.unpool5 = nn.MaxUnpool2d(kernel_size=(3, 2), stride=2, padding=0)
        # 20 x 11
        self.deconv53 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 20 x 11
        self.deconv52 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 20 x 11
        self.deconv51 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 20 x 11

        # 20 x 11
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 40 x 22
        self.deconv43 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 40 x 22
        self.deconv42 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 40 x 22
        self.deconv41 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 40 x 22

        # 40 x 22
        self.unpool3 = nn.MaxUnpool2d(kernel_size=(3, 2), stride=2, padding=0)
        # 80 x 45
        self.deconv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.deconv31 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 80 x 45

        # 80 x 45
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 160 x 90
        self.deconv22 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 160 x 90
        self.deconv21 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 160 x 90

        # 160 x 90
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 320 x 180
        self.deconv12 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 320 x 180
        self.deconv11 = nn.Sequential(
            nn.Conv2d(kernel[4], 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(2),
        )
        # 320 x 180

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(kernel[0], 6)

    def forward(self, x):

        x = self.conv11(x)
        x = self.relu(x)
        id1 = x
        x = self.conv12(x)
        x += id1
        x = self.relu(x)
        x, idx1 = self.pool1(x)

        x = self.conv21(x)
        x = self.relu(x)
        id2 = x
        x = self.conv22(x)
        x += id2
        x = self.relu(x)
        x, idx2 = self.pool2(x)

        x = self.conv31(x)
        x = self.relu(x)
        id3 = x
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x += id3
        x = self.relu(x)
        x, idx3 = self.pool3(x)

        x = self.conv41(x)
        x = self.relu(x)
        id4 = x
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x += id4
        x = self.relu(x)
        x, idx4 = self.pool4(x)

        x = self.conv51(x)
        x = self.relu(x)
        id5 = x
        x = self.conv52(x)
        x = self.relu(x)
        x = self.conv53(x)
        x += id5
        x = self.relu(x)
        x, idx5 = self.pool5(x)

        y = self.avgpool(x)
        y = self.fc(y.squeeze())

        x = self.unpool5(x, indices=idx5)
        uid5 = x
        x = self.deconv53(x)
        x = self.relu(x)
        x = self.deconv52(x)
        x += uid5
        x = self.relu(x)
        x = self.deconv51(x)
        x = self.relu(x)

        x = self.unpool4(x, indices=idx4)
        uid4 = x
        x = self.deconv43(x)
        x = self.relu(x)
        x = self.deconv42(x)
        x += uid4
        x = self.relu(x)
        x = self.deconv41(x)
        x = self.relu(x)

        x = self.unpool3(x, indices=idx3)
        uid3 = x
        x = self.deconv32(x)
        x += uid3
        x = self.relu(x)
        x = self.deconv31(x)
        x = self.relu(x)

        x = self.unpool2(x, indices=idx2)
        uid2 = x
        x = self.deconv22(x)
        x += uid2
        x = self.relu(x)
        x = self.deconv21(x)
        x = self.relu(x)

        x = self.unpool1(x, indices=idx1)
        uid1 = x
        x = self.deconv12(x)
        x += uid1
        x = self.relu(x)
        x = self.deconv11(x)
        x = self.relu(x)

        return y, x


#
class MySegNet3(nn.Module):

    def __init__(self, num_classes=21):
        super(MySegNet3, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        kernel = [16, 32, 64, 128, 128]

        # 320 x 180
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.conv12 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 160 x 90
        self.conv21 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.conv22 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 80 x 45
        self.conv31 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv33 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1, return_indices=True)
        # 20 x 15
        self.conv41 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.conv42 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.conv43 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1, return_indices=True)
        # 5 x 5

        kernel = [128, 128, 64, 32, 16]

        # 5 x 5
        self.unpool4 = nn.MaxUnpool2d(kernel_size=(3, 4), stride=(3, 4), padding=0)
        # 20 x 15
        self.deconv43 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 20 x 15
        self.deconv42 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 20 x 15
        self.deconv41 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 20 x 15
        self.unpool3 = nn.MaxUnpool2d(kernel_size=(3, 4), stride=(3, 4), padding=0)
        # 80 x 45
        self.deconv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.deconv31 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 80 x 45
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 160 x 90
        self.deconv22 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 160 x 90
        self.deconv21 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 160 x 90
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 320 x 180
        self.deconv12 = nn.Sequential(
            nn.Conv2d(kernel[4], kernel[4], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[4]),
        )
        # 320 x 180
        self.deconv11 = nn.Sequential(
            nn.Conv2d(kernel[4], 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(2),
        )
        # 320 x 180

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(kernel[0], 6)

    def forward(self, x):

        x = self.conv11(x)
        x = self.relu(x)
        id1 = x
        x = self.conv12(x)
        x += id1
        x = self.relu(x)
        x, idx1 = self.pool1(x)

        x = self.conv21(x)
        x = self.relu(x)
        id2 = x
        x = self.conv22(x)
        x += id2
        x = self.relu(x)
        x, idx2 = self.pool2(x)

        x = self.conv31(x)
        x = self.relu(x)
        id3 = x
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x += id3
        x = self.relu(x)
        x, idx3 = self.pool3(x)

        x = self.conv41(x)
        x = self.relu(x)
        id4 = x
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x += id4
        x = self.relu(x)
        x, idx4 = self.pool4(x)

        # x = self.conv51(x)
        # x = self.relu(x)
        # id5 = x
        # x = self.conv52(x)
        # x = self.relu(x)
        # x = self.conv53(x)
        # x += id5
        # x = self.relu(x)
        # x, idx5 = self.pool5(x)

        y = self.avgpool(x)
        y = self.fc(y.squeeze())

        # x = self.unpool5(x, indices=idx5)
        # uid5 = x
        # x = self.deconv53(x)
        # x = self.relu(x)
        # x = self.deconv52(x)
        # x += uid5
        # x = self.relu(x)
        # x = self.deconv51(x)
        # x = self.relu(x)

        x = self.unpool4(x, indices=idx4)
        uid4 = x
        x = self.deconv43(x)
        x = self.relu(x)
        x = self.deconv42(x)
        x += uid4
        x = self.relu(x)
        x = self.deconv41(x)
        x = self.relu(x)

        x = self.unpool3(x, indices=idx3)
        uid3 = x
        x = self.deconv32(x)
        x += uid3
        x = self.relu(x)
        x = self.deconv31(x)
        x = self.relu(x)

        x = self.unpool2(x, indices=idx2)
        uid2 = x
        x = self.deconv22(x)
        x += uid2
        x = self.relu(x)
        x = self.deconv21(x)
        x = self.relu(x)

        x = self.unpool1(x, indices=idx1)
        uid1 = x
        x = self.deconv12(x)
        x += uid1
        x = self.relu(x)
        x = self.deconv11(x)
        x = self.relu(x)

        return y, x


#
class MySegNet4(nn.Module):

    def __init__(self, num_classes=21):
        super(MySegNet4, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        kernel = [16, 32, 64, 128]

        # 320 x 180
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.conv12 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 320 x 180
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 160 x 90
        self.conv21 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.conv22 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 160 x 90
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        # 80 x 45
        self.conv31 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv32 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.conv33 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1, return_indices=True)
        # 20 x 15
        self.conv41 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.conv42 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.conv43 = nn.Sequential(
            nn.Conv2d(kernel[3], kernel[3], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[3]),
        )
        # 20 x 15
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 4), stride=(3, 4), padding=0, dilation=1, return_indices=True)
        # 5 x 5

        kernel = [128, 64, 32]

        # 5 x 5
        self.unpool3 = nn.MaxUnpool2d(kernel_size=(3, 4), stride=(3, 4), padding=0)
        # 20 x 15
        self.deconv32 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[0], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[0]),
        )
        # 20 x 15
        self.deconv31 = nn.Sequential(
            nn.Conv2d(kernel[0], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 20 x 15
        self.unpool2 = nn.MaxUnpool2d(kernel_size=(3, 4), stride=(3, 4), padding=0)
        # 80 x 45
        self.deconv22 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[1]),
        )
        # 80 x 45
        self.deconv21 = nn.Sequential(
            nn.Conv2d(kernel[1], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 80 x 45
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        # 160 x 90
        self.deconv12 = nn.Sequential(
            nn.Conv2d(kernel[2], kernel[2], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(kernel[2]),
        )
        # 160 x 90
        self.deconv11 = nn.Sequential(
            nn.Conv2d(kernel[2], 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(2),
        )
        # 160 x 90

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(kernel[0], 6)

    def forward(self, x):

        x = self.conv11(x)
        x = self.relu(x)
        id1 = x
        x = self.conv12(x)
        x += id1
        x = self.relu(x)
        x, idx1 = self.pool1(x)

        x = self.conv21(x)
        x = self.relu(x)
        id2 = x
        x = self.conv22(x)
        x += id2
        x = self.relu(x)
        x, idx2 = self.pool2(x)

        x = self.conv31(x)
        x = self.relu(x)
        id3 = x
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x += id3
        x = self.relu(x)
        x, idx3 = self.pool3(x)

        x = self.conv41(x)
        x = self.relu(x)
        id4 = x
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x += id4
        x = self.relu(x)
        x, idx4 = self.pool4(x)

        y = self.avgpool(x)
        y = self.fc(y.squeeze())

        x = self.unpool3(x, indices=idx4)
        uid3 = x
        x = self.deconv32(x)
        x += uid3
        x = self.relu(x)
        x = self.deconv31(x)
        x = self.relu(x)

        x = self.unpool2(x, indices=idx3)
        uid2 = x
        x = self.deconv22(x)
        x += uid2
        x = self.relu(x)
        x = self.deconv21(x)
        x = self.relu(x)

        x = self.unpool1(x, indices=idx2)
        uid1 = x
        x = self.deconv12(x)
        x += uid1
        x = self.relu(x)
        x = self.deconv11(x)
        x = self.relu(x)

        return y, x


class MultiHead(nn.Module):

    def __init__(self):
        super(MultiHead, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 320 x 180
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 3), stride=(5, 1), padding=(3, 1), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 320 x 36
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 11), stride=(1, 5), padding=(1, 3), dilation=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # 64 x 180
        self.pool1_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0, dilation=1)
        # 64 x 36
        self.pool1_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1), padding=0, dilation=1)
        # 64 x 36
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 64 x 36
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
        )
        # 64 x 36
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 32 x 18
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 32 x 18
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
        )
        # 32 x 18
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 16 x 9
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )
        # 16 x 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
        )
        # 16 x 9
        self.pool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 1 x 1
        self.fc_1 = nn.Linear(512, 2)
        self.fc_2 = nn.Linear(512, 2)
        self.fc_3 = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.cat((self.pool1_w(self.relu(self.conv1_h(x))), self.pool1_h(self.relu(self.conv1_w(x)))), dim=1)
        id1 = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x += id1
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        id2 = x
        x = self.relu(x)
        x = self.conv3_2(x)
        x += id2
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        id3 = x
        x = self.relu(x)
        x = self.conv4_2(x)
        x += id3
        x = self.relu(x)
        x = self.pool4(x)
        x = torch.flatten(x, 1)
        return self.fc_1(x), self.fc_2(x), self.fc_3(x)

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

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=6):
        super(ResNet, self).__init__()

        features = [32, 64, 128, 256]

        self.inplanes = features[0]

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # in -> 320 x 180 | out -> 320 x 36
        self.conv1_h = nn.Sequential(
            nn.Conv2d(3, self.inplanes // 2, kernel_size=(11, 1), stride=(5, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(self.inplanes // 2)
        )

        # in -> 320 x 180 | out -> 64 x 180
        self.conv1_w = nn.Sequential(
            nn.Conv2d(3, self.inplanes // 2, kernel_size=(1, 11), stride=(1, 5), padding=(0, 3), bias=False),
            nn.BatchNorm2d(self.inplanes // 2)
        )

        self.conv1_hw = nn.Sequential(
            nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=(1, 11), stride=(1, 5), padding=(0, 3), bias=False),
            nn.BatchNorm2d(self.inplanes // 2)
        )
        self.conv1_wh = nn.Sequential(
            nn.Conv2d(self.inplanes // 2, self.inplanes // 2, kernel_size=(11, 1), stride=(5, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(self.inplanes // 2)
        )

        self.relu = nn.ReLU(inplace=True)

        # in -> 64 x 180 | out -> 64 x 36
        # self.maxpool_h = nn.MaxPool2d(kernel_size=(5, 1), stride=(5, 1))

        # in -> 320 x 36 | out -> 64 x 36
        # self.maxpool_w = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))

        self.layer1 = self._make_layer(block, features[0], layers[0])
        self.layer2 = self._make_layer(block, features[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, features[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, features[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # hw = self.conv1_h(x)
        # hw = self.relu(hw)
        # hw = self.maxpool_w(hw)
        #
        # wh = self.conv1_w(x)
        # wh = self.relu(wh)
        # wh = self.maxpool_h(wh)
        #
        # x = torch.cat((hw, wh), dim=1)

        hw = self.conv1_h(x)
        hw = self.relu(hw)
        hw = self.conv1_hw(hw)

        wh = self.conv1_w(x)
        wh = self.relu(wh)
        wh = self.conv1_wh(wh)

        x = torch.cat((hw, wh), dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, channels, decode=True):
        super(SegNet, self).__init__()
        if channels is None:
            channels = [64, 128, 256, 512, 512]

        self.decode = decode
        self.relu = nn.ReLU(inplace=True)

        blocks = [2, 2, 3, 3, 3]
        kernel = [3] + channels

        for i in range(len(blocks)):
            for j in range(blocks[i]):
                block = []
                if j == 0:
                    block.append(nn.Conv2d(kernel[i], kernel[i + 1], kernel_size=3, padding=1))
                else:
                    block.append(nn.Conv2d(kernel[i + 1], kernel[i + 1], kernel_size=3, padding=1))
                block.append(nn.BatchNorm2d(kernel[i + 1]))
                setattr(self, 'conv' + str(i + 1) + str(j + 1), nn.Sequential(*block))
            if i == 2 or i == 4:
                setattr(self, 'pool' + str(i + 1), nn.MaxPool2d(kernel_size=(3, 2), stride=2, return_indices=True))
            else:
                setattr(self, 'pool' + str(i + 1), nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(kernel[-1], 6)

        if self.decode:
            channels.reverse()
            blocks.reverse()

            kernel = channels + [2]

            for i in range(len(blocks)):
                for j in range(blocks[i]):
                    block = []
                    if j != blocks[i] - 1:
                        block.append(nn.Conv2d(kernel[i], kernel[i], kernel_size=3, padding=1))
                        block.append(nn.BatchNorm2d(kernel[i]))
                    else:
                        block.append(nn.Conv2d(kernel[i], kernel[i + 1], kernel_size=3, padding=1))
                        block.append(nn.BatchNorm2d(kernel[i + 1]))
                    setattr(self, 'deconv' + str(len(blocks) - i) + str(blocks[i] - j), nn.Sequential(*block))
                if i == 0 or i == 2:
                    setattr(self, 'unpool' + str(len(blocks) - i), nn.MaxUnpool2d(kernel_size=(3, 2), stride=2))
                else:
                    setattr(self, 'unpool' + str(len(blocks) - i), nn.MaxUnpool2d(kernel_size=2, stride=2))

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

        if self.decode:
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
        else:
            return y

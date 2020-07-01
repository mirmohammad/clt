import os
import random

import pandas as pd
from PIL import Image, ImageDraw
from torch.utils import data
from torchvision.transforms import functional as tf

from utils import draw


class CLT(data.Dataset):

    def __init__(self, root_dir, cows, vflip=False, hflip=False, transform=None):
        self.vflip = vflip
        self.hflip = hflip
        self.transform = transform

        # self.grid_w = [x * 5 for x in range(1, 64)]
        # self.grid_h = [x * 5 for x in range(1, 36)]

        images = [os.path.join(root_dir, 'images', x) for x in cows]
        images = [sorted([os.path.join(x, y) for y in os.listdir(x)]) for x in images]
        images = [x for cow in images for x in cow]

        labels = [os.path.join(root_dir, 'labels', x + '.csv') for x in cows]
        labels = [pd.read_csv(x, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3']).values.astype('float32') for x in labels]
        labels = [x / 4. for cow in labels for x in cow]

        self.dataset = list(zip(images, labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = Image.open(image)

        # draw = ImageDraw.Draw(image)
        # for x in self.grid_w:
        #     draw.line([x, 0, x, 179], fill=128, width=0)
        #
        # for y in self.grid_h:
        #     draw.line([0, y, 319, y], fill=128, width=0)

        # label = label / 4.
        # label[1] += 70
        # label[3] += 70
        # label[5] += 70
        trngl = draw.get_triangle(label)

        if self.transform:
            # image = tf.pad(image, (0, 70))
            # trisg = tf.pad(trisg, (0, 70))
            # if self.vflip:
            #     if random.random() > 0.5:
            #         image = tf.vflip(image)
            #         # trisg = tf.vflip(trisg)
            #         label[1] = 179 - label[1]
            #         label[3] = 179 - label[3]
            #         label[5] = 179 - label[5]
            # if self.hflip:
            #     if random.random() > 0.5:
            #         image = tf.hflip(image)
            #         # trisg = tf.vflip(trisg)
            #         label[0] = 319 - label[0]
            #         label[2] = 319 - label[2]
            #         label[4] = 319 - label[4]
            image = self.transform(image)
            trngl = self.transform(trngl)

        return image, label, trngl.long()

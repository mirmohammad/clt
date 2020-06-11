import os
import random

import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import functional as tf

from utils import draw


class CLT(data.Dataset):

    def __init__(self, root_dir, cows, transform=None):
        self.transform = transform

        images = [os.path.join(root_dir, 'images', x) for x in cows]
        images = [sorted([os.path.join(x, y) for y in os.listdir(x)]) for x in images]
        images = [x for cow in images for x in cow]

        labels = [os.path.join(root_dir, 'labels', x + '.csv') for x in cows]
        labels = [pd.read_csv(x, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3']).values for x in labels]
        labels = [x for cow in labels for x in cow]

        self.dataset = list(zip(images, labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = Image.open(image)
        label = label // 4
        label[1] += 70
        label[3] += 70
        label[5] += 70
        trisg = draw.get_triangle(label)

        if self.transform:
            image = tf.pad(image, (0, 70))
            trisg = tf.pad(trisg, (0, 70))
            if random.random() > 0.5:
                image = tf.vflip(image)
                trisg = tf.vflip(trisg)
                label[1] = 319 - label[1]
                label[3] = 319 - label[3]
                label[5] = 319 - label[5]
            image = self.transform(image)
            trisg = self.transform(trisg)

        return image, label.astype('float32'), trisg

import math
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def distance(points):
    p1_p2 = math.sqrt((points[2] - points[0]) ** 2 + (points[3] - points[1]) ** 2)
    p1_p3 = math.sqrt((points[4] - points[0]) ** 2 + (points[5] - points[1]) ** 2)
    p2_p3 = math.sqrt((points[4] - points[2]) ** 2 + (points[5] - points[3]) ** 2)
    return np.array([p1_p2, p1_p3, p2_p3])


def distance_angle(points):
    d_p1_p2 = math.sqrt((points[2] - points[0]) ** 2 + (points[3] - points[1]) ** 2)
    d_p1_p3 = math.sqrt((points[4] - points[0]) ** 2 + (points[5] - points[1]) ** 2)
    d_p2_p3 = math.sqrt((points[4] - points[2]) ** 2 + (points[5] - points[3]) ** 2)
    a_p1_p2 = math.asin(abs(points[0] - points[2]) / d_p1_p2)
    a_p1_p3 = math.acos(abs(points[0] - points[4]) / d_p1_p3)
    a_p2_p3 = math.acos(abs(points[2] - points[4]) / d_p2_p3)
    return np.array([d_p1_p2, a_p1_p2, d_p1_p3, a_p1_p3, d_p2_p3, a_p2_p3])


class CltDataset(Dataset):

    def __init__(self, root, cows, sub=None, aug_mode='normal', transform=None):
        self.aug_mode = aug_mode
        self.transform = transform
        self.df = pd.DataFrame(columns=['image', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])

        for cow in cows:
            image_dir = root + '/images/' + cow
            label_dir = root + '/tags/' + cow + '.csv'

            image_name = sorted(os.listdir(image_dir))
            image_path = [image_dir + '/' + s for s in image_name]

            df = pd.read_csv(label_dir, names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
            df.insert(loc=0, column='image', value=pd.Series(image_path))

            if sub:
                idx = int(df.shape[0] * sub)
                df = df.iloc[:idx]

            self.df = self.df.append(df, ignore_index=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        image = Image.open(sample['image'])
        label = sample['x1':'y3'].values

        if self.aug_mode == 'normal':
            label = label.astype('float32').reshape(6)
        elif self.aug_mode == 'distance':
            label = np.append(label, distance(label))
            label = label.astype('float32').reshape(9)
        elif self.aug_mode == 'distance_angle':
            # da = distance_angle(label)
            # da = np.insert(da, 0, label[0:2])
            # da = np.insert(da, 4, label[2:4])
            # label = np.insert(da, 8, label[4:6])
            label = np.append(label, distance_angle(label))
            label = label.astype('float32').reshape(12)

        if self.transform:
            image = self.transform(image)

        return image, label

import argparse
import os

import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import CLT

parser = argparse.ArgumentParser(description="CLT | Cow's Location Tracking Project (ETS/McGill)")
parser.add_argument('dir', help='path to the dataset directory containing images and labels')
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()

root_dir = args.dir
batch_size = args.batch_size
num_workers = args.num_workers

images_dir = os.path.join(root_dir, 'images')
cows = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
cows = sorted([os.path.basename(x) for x in cows if os.path.isdir(x)])

dataset = CLT(root_dir=root_dir, cows=cows, decode=False, transform=transforms.ToTensor())
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

run_std = torch.zeros(3)
run_mean = torch.zeros(3)
run_batch = 0

monitor = tqdm(loader, desc='norm')
for img, _ in monitor:
    std, mean = torch.std_mean(img, dim=[0, 2, 3])
    run_std += std
    run_mean += mean
    run_batch += 1

print(run_std / run_batch)
print(run_mean / run_batch)

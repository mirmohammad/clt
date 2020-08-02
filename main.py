import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import CLT
from model import segnet

parser = argparse.ArgumentParser(description="CLT | Cow's Location Tracking Project (ETS/McGill)")
# Path arguments
parser.add_argument('dir', help='path to the dataset directory containing images and labels')
parser.add_argument('--logs_dir', default='.', help='path to directory to store logs')
# Training arguments
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--aux_ratio', default=0.4, type=float)
parser.add_argument('--mse', action='store_true')
parser.add_argument('--mae', action='store_true')
# Optimizer arguments
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
# Experiment arguments
parser.add_argument('--random_seed', default=42)
parser.add_argument('--manual_seed', action='store_true')
parser.add_argument('--gpu', default='0', type=str)
# Model arguments
parser.add_argument('--decode', action='store_true')
parser.add_argument('--channels', nargs='+', type=int)
# Augmentation arguments
parser.add_argument('--vflip', action='store_true')
parser.add_argument('--hflip', action='store_true')
# Scheduler arguments
parser.add_argument('--step_lr', action='store_true')
parser.add_argument('--multi_lr', action='store_true')
parser.add_argument('--skd_step', default=25, type=int)
parser.add_argument('--skd_gamma', default=0.1, type=float)
parser.add_argument('--skd_mile', nargs='+', type=int)
# Fixed arguments
parser.add_argument('--out_size', default=6, type=int)
parser.add_argument('--num_workers', default=8, type=int)

args = parser.parse_args()

cuda = torch.cuda.is_available()

# Path arguments
root_dir = args.dir
logs_dir = args.logs_dir
# Training arguments
batch_size = args.batch_size
num_epochs = args.num_epochs
aux_ratio = args.aux_ratio
mse = args.mse
mae = args.mae
# Optimizer arguments
learning_rate = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay
# Experiment arguments
random_seed = args.random_seed
manual_seed = args.manual_seed
gpu = args.gpu
# Model arguments
decode = args.decode
channels = args.channels
# Augmentation arguments
vflip = args.vflip
hflip = args.hflip
# Scheduler arguments
step_lr = args.step_lr
multi_lr = args.multi_lr
skd_step = args.skd_step
skd_gamma = args.skd_gamma
skd_mile = args.skd_mile
# Fixed arguments
out_size = args.out_size
num_workers = args.num_workers

if manual_seed:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

device = torch.device('cuda:' + gpu if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

assert step_lr != multi_lr, 'Must choose only one scheduler method (Options: --step_lr, --multi_lr)'

images_dir = os.path.join(root_dir, 'images')
cows = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
cows = sorted([os.path.basename(x) for x in cows if os.path.isdir(x)])

train_cows = cows[1:]
valid_cows = cows[:1]

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.4066, 0.4085, 0.4028], std=[0.1809, 0.1871, 0.1975])
])

train_dataset = CLT(root_dir=root_dir, cows=train_cows, decode=decode, scale=1, transform=transform)
valid_dataset = CLT(root_dir=root_dir, cows=valid_cows, decode=decode, scale=1, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if mae:
    out_criterion = nn.L1Loss()
elif mse:
    out_criterion = nn.MSELoss()
else:
    out_criterion = nn.SmoothL1Loss()

seg_criterion = nn.CrossEntropyLoss()
model = segnet.SegNet(channels=channels, decode=decode).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

if step_lr:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=skd_step, gamma=skd_gamma)
if multi_lr:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=skd_mile, gamma=skd_gamma)


def iterate(ep, mode):
    if mode == 'train':
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = valid_loader

    num_samples = 0
    run_loss = 0.
    run_err = torch.zeros(3)

    monitor = tqdm(loader, desc=mode)
    for it in monitor:
        if decode:
            img, lbl, tri = it
            out, seg = model(img.to(device))
            out_loss = out_criterion(out, lbl.to(device))
            seg_loss = seg_criterion(seg, tri.squeeze(1).to(device))
            loss = out_loss + aux_ratio * seg_loss
        else:
            img, lbl = it
            out = model(img.to(device))
            loss = out_criterion(out, lbl.to(device))

        num_samples += lbl.size(0)
        run_loss += loss.item() * lbl.size(0)
        run_err += ((out.detach().cpu() - lbl) ** 2).view(-1, 3, 2).sum(2).sqrt().sum(0)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        monitor.set_postfix(epoch=ep, loss=run_loss / num_samples, err=(run_err / num_samples).round().tolist(), avg=run_err.mean().item() / num_samples)

    if mode == 'train':
        scheduler.step()

    return run_err / num_samples


if __name__ == '__main__':
    best_avg = 1e16
    best_ep = -1
    for epoch in range(num_epochs):
        err = iterate(epoch, 'train')
        tqdm.write(f'Train | Epoch {epoch} | Error {err.tolist()}')
        with torch.no_grad():
            err = iterate(epoch, 'valid')
            if err.mean() <= best_avg:
                tqdm.write(f'NEW BEST VALIDATION | New Average {err.mean()} | Improvement {best_avg - err.mean()}')
                best_avg = err.mean()
                best_ep = epoch
                torch.save(model.state_dict(), '2_params_' + str(err.mean().item()) + '_' + str(err) + '_.pt')
            tqdm.write(f'Valid | Epoch {epoch} | Error {err.tolist()} | Best Average {best_avg} | Best Epoch {best_ep}')

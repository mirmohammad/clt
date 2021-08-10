import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from datetime import datetime
from torch.utils import data
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from data import CLT

parser = argparse.ArgumentParser(description="CLT | Cow's Location Tracking Project (ETS/McGill)")
parser.add_argument('dir', help='path to yaml configuration file')
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()
cuda = torch.cuda.is_available()

cfg = yaml.safe_load(open(args.dir))

# TODO: CHANGE 'model' AND 'loss' HERE FOR LOGGING
log_dir = os.path.join(cfg['log_dir'], 'resnet18', 'SmoothL1Loss', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(log_dir, exist_ok=True)

if cfg['manual_seed']:
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])

device = torch.device('cuda:' + args.gpu if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

images_dir = os.path.join(cfg['data_dir'], 'images')
cows = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
cows = sorted([os.path.basename(x) for x in cows if os.path.isdir(x)])

train_cows = cows[1:]
valid_cows = cows[:1]

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.4066, 0.4085, 0.4028], std=[0.1809, 0.1871, 0.1975])
])

train_dataset = CLT(root_dir=cfg['data_dir'], cows=train_cows, decode=cfg['decode'], scale=1, transform=transform)
valid_dataset = CLT(root_dir=cfg['data_dir'], cows=valid_cows, decode=cfg['decode'], scale=1, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], drop_last=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

out_criterion = nn.SmoothL1Loss()
seg_criterion = nn.CrossEntropyLoss()

# model = segnet.SegNet(channels=cfg['channels'], decode=cfg['decode']).to(device)
model = models.resnet18(pretrained=False, num_classes=cfg['num_classes']).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step'], gamma=cfg['gamma'])


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
        if cfg['decode']:
            img, lbl, tri = it
            out, seg = model(img.to(device))
            out_loss = out_criterion(out, lbl.to(device))
            seg_loss = seg_criterion(seg, tri.squeeze(1).to(device))
            loss = out_loss + cfg['aux_ratio'] * seg_loss
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

    return run_loss / num_samples, run_err / num_samples


if __name__ == '__main__':
    best_avg = 1e16
    best_ep = -1

    train_loss = []
    train_acc = []

    valid_loss = []
    valid_acc = []

    for epoch in range(cfg['num_epochs']):
        l, err = iterate(epoch, 'train')
        train_loss.append(l)
        train_acc.append(err.mean())
        tqdm.write(f'Train | Epoch {epoch} | Error {err.tolist()}')
        with torch.no_grad():
            l, err = iterate(epoch, 'valid')
            valid_loss.append(l)
            valid_acc.append(err.mean())
            if err.mean() <= best_avg:
                tqdm.write(f'NEW BEST VALIDATION | New Average {err.mean()} | Improvement {best_avg - err.mean()}')
                best_avg = err.mean()
                best_ep = epoch
                torch.save(model.state_dict(), os.path.join(log_dir, '_' + str(err.mean().item()) + '_' + str(err.tolist()) + '_' + '.pt'))
            tqdm.write(f'Valid | Epoch {epoch} | Error {err.tolist()} | Best Average {best_avg} | Best Epoch {best_ep}')

    np.savetxt(os.path.join(log_dir, 'train_loss.csv'), np.asarray(train_loss), delimiter=',')
    np.savetxt(os.path.join(log_dir, 'train_acc.csv'), np.asarray(train_acc), delimiter=',')
    np.savetxt(os.path.join(log_dir, 'valid_loss.csv'), np.asarray(valid_loss), delimiter=',')
    np.savetxt(os.path.join(log_dir, 'valid_acc.csv'), np.asarray(valid_acc), delimiter=',')

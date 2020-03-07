import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from CltDataset import CltDataset

# from resnet import *

# ------------------------------------ CUDA Setup ----------------------------------------------------------------------

cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

# ------------------------------------ Dataset Configuration -----------------------------------------------------------

cows = ['419-IS', '442-IS', '5160-IS', '5207-IS', '5246-IS', '5254-IS', '5258-IS', '5259-IS', '5269-IS',
        '5275-IS', '5278-IS', '5279-IS', '5283-IS', '5294-IS', '5296-IS', '5298-IS', '5300-IS', '5301-IS',
        '5302-IS', '5308-IS', '5322-IS', '6294-IS', '7097-IS', '9011-IS']

file_server = '/export/livia/data/CommonData/cows_data/clt'
local_storage = '/state/data/vision/msaadati/chain'

train_cows = cows[1:]
valid_cows = cows[:1]

aug_modes = {'normal': 6, 'distance': 9, 'distance_angle': 12}
aug_mode = 'normal'

sub = None
scale = 4
root = file_server + '/final/s' + str(scale)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CltDataset(root=root, cows=train_cows, sub=sub, aug_mode=aug_mode, transform=transform)
valid_dataset = CltDataset(root=root, cows=valid_cows, sub=sub, aug_mode=aug_mode, transform=transform)

# ------------------------------------ Data Loader Configuration -------------------------------------------------------

batch_size = 64
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ------------------------------------ Model Configuration -------------------------------------------------------------

classes = aug_modes[aug_mode]

params = {'num_classes': classes}
model = models.resnet18(pretrained=False, **params)

# model.load_state_dict(torch.load('params.pt'))
model = model.to(device)

tqdm.write('number of model parameters: %i' % sum([p.numel() for p in model.parameters()]))

# ------------------------------------ Optimizer Configuration ---------------------------------------------------------

momentum = 0.9
weight_decay = 5e-4

epochs = 100
learning_rate = 1e-3
milestones = [40, 80]
# eta_min = 1e-6
gamma = 0.1

criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

experiment = '/resnet18/419-IS/baseline/multi_step/RMSprop/MSE/SUB_NONE-SCALE_4-BATCH_64-RUN_0'


# ------------------------------------ Main ----------------------------------------------------------------------------

def iterate(ep, mode, loader):
    if mode == 'train':
        scheduler.step()
        model.train()
    elif mode == 'valid':
        model.eval()

    num_images = 0
    run_loss = 0.
    run_pe = torch.zeros([3], dtype=torch.float32).to(device)
    run_ape = 0.

    monitor = tqdm(loader, desc=mode)
    for (images, labels) in monitor:
        images, labels = images.to(device), labels.to(device)

        labels /= scale

        outputs = model(images)
        loss = criterion(outputs, labels)

        num_images += images.size(0)
        run_loss += loss.item() * images.size(0)

        with torch.no_grad():
            if aug_mode != 'normal':
                outputs = outputs[:, 0:6]
                labels = labels[:, 0:6]
            pe = ((outputs - labels) ** 2).view(-1, 3, 2).sum(2).sqrt().sum(0)
            run_pe += pe
            run_ape += (pe.sum() / 3).item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        monitor.set_postfix(epoch=ep, loss=run_loss / num_images, pe=torch.round(run_pe / num_images).tolist(),
                            ape=run_ape / num_images)

    return run_loss / num_images, run_ape / num_images, run_pe.cpu().numpy() / num_images


def log_results(file_name, l, ape, pe):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w+')
    i = 1
    for x, y, z in zip(l, ape, pe):
        file.write('{:d},'.format(i))
        file.write('{:.5f},'.format(x))
        file.write('{:.5f},'.format(y))
        file.write('{:.5f},'.format(z[0]))
        file.write('{:.5f},'.format(z[1]))
        file.write('{:.5f}'.format(z[2]))
        file.write('\n')
        i = i + 1
    file.close()


if __name__ == '__main__':

    best_ape = 1e50

    t_l = []
    t_ape = []
    t_pe = []

    v_l = []
    v_ape = []
    v_pe = []

    for epoch in range(epochs):
        train_l, train_ape, train_pe = iterate(epoch, 'train', train_loader)
        t_l.append(train_l)
        t_ape.append(train_ape)
        t_pe.append(train_pe)

        with torch.no_grad():
            valid_l, valid_ape, valid_pe = iterate(epoch, 'valid', valid_loader)
            v_l.append(valid_l)
            v_ape.append(valid_ape)
            v_pe.append(valid_pe)

            if valid_ape < best_ape and epoch >= 10:
                best_ape = valid_ape
                torch.save(model.state_dict(), file_server + experiment + '/params.pt')

        tqdm.write('')

        log_results(file_server + experiment + '/TRAIN.csv', t_l, t_ape, t_pe)
        log_results(file_server + experiment + '/VALID.csv', v_l, v_ape, v_pe)

import os

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from data import CLT

# ------------------------------------ CUDA Setup ----------------------------------------------------------------------

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

# ------------------------------------ Dataset Configuration -----------------------------------------------------------

cows = ['419-IS', '442-IS', '5160-IS', '5207-IS', '5246-IS', '5254-IS', '5258-IS', '5259-IS', '5269-IS',
        '5275-IS', '5278-IS', '5279-IS', '5283-IS', '5294-IS', '5296-IS', '5298-IS', '5300-IS', '5301-IS',
        '5302-IS', '5308-IS', '5322-IS', '6294-IS', '7097-IS', '9011-IS']

file_server = '/export/livia/data/CommonData/cows_data/clt'
local_storage = '/state/data/vision/msaadati/chain'

aug_modes = {'normal': 6, 'distance': 9, 'distance_angle': 12}
aug_mode = 'distance_angle'

sub = None
scale = 4
root = local_storage + '/final/s' + str(scale)

transform = transforms.Compose([transforms.ToTensor()])

extract_datasets = [CltDataset(root=root, cows=[cows[i]], sub=sub, aug_mode=aug_mode, transform=transform) for i in
                    range(len(cows))]

# ------------------------------------ Data Loader Configuration -------------------------------------------------------

batch_size = 256
num_workers = 8

extract_loaders = [DataLoader(extract_datasets[i], batch_size=batch_size, shuffle=False, num_workers=num_workers) for i
                   in range(len(cows))]

# ------------------------------------ Model Configuration -------------------------------------------------------------

experiment = '/resnet18/419-IS/distance_angle/cosine/RMSprop/MSE/SUB_NONE-SCALE_4-BATCH_256-RUN_0'

classes = aug_modes[aug_mode]

params = {'num_classes': classes}
# model = resnet18(pretrained=False, **params)
model = models.resnet18(pretrained=False, **params)
# torch.save(model.state_dict(), file_server + experiment + '/params.pt')
model.load_state_dict(torch.load(file_server + experiment + '/params.pt'))
model = model.to(device)

tqdm.write('number of model parameters: %i' % sum([p.numel() for p in model.parameters()]))


# ------------------------------------ Main ----------------------------------------------------------------------------

def iterate(cow, loader):
    model.eval()

    num_images = 0
    run_pe = torch.zeros([3], dtype=torch.float32).to(device)
    run_pe_all = []
    run_ape = 0.

    monitor = tqdm(loader, desc='extract')
    for (images, labels) in monitor:
        images, labels = images.to(device), labels.to(device)

        labels /= scale

        outputs = model(images)

        num_images += images.size(0)

        with torch.no_grad():
            if aug_mode != 'normal':
                outputs = outputs[:, 0:6]
                labels = labels[:, 0:6]
            pe = ((outputs - labels) ** 2).view(-1, 3, 2).sum(2).sqrt()
            run_pe_all.append(pe)
            run_pe += pe.sum(0)
            run_ape += (pe.sum() / 3).item()

        monitor.set_postfix(cow=cows[cow], pe=torch.round(run_pe / num_images).tolist(), ape=run_ape / num_images)

    return run_ape / num_images, run_pe_all, run_pe.cpu().numpy() / num_images


def log_results(file_name, ape, pe_all, pe):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w+')
    file.write('Average Pixel Error for all of the images of this cow,d1,d2,d3\n')
    file.write('{:.5f},'.format(ape))
    file.write('{:.5f},'.format(pe[0]))
    file.write('{:.5f},'.format(pe[1]))
    file.write('{:.5f}'.format(pe[2]))
    file.write('\n')
    file.write('Pixel Error values for each individual image (ordered by file names)\n')
    file.write('d1,d2,d3\n')
    for t in pe_all:
        t = t.cpu().numpy()
        for b in t:
            file.write('{:.5f},'.format(b[0]))
            file.write('{:.5f},'.format(b[1]))
            file.write('{:.5f}'.format(b[2]))
            file.write('\n')
    file.close()


def log_all_cows(file_name, ape, pe):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w+')
    file.write('Cow,Average Pixel Error,Pixel Error d1,Pixel Error d2,Pixel Error d3\n')
    for x, y, z in zip(cows, ape, pe):
        file.write(x + ',')
        file.write('{:.5f},'.format(y))
        file.write('{:.5f},'.format(z[0]))
        file.write('{:.5f},'.format(z[1]))
        file.write('{:.5f}'.format(z[2]))
        file.write('\n')
    file.close()


if __name__ == '__main__':

    best_ape = 1e50

    e_ape = []
    e_pe = []

    for i in range(len(cows)):
        with torch.no_grad():
            ext_ape, ext_pe_all, ext_pe = iterate(i, extract_loaders[i])
            e_ape.append(ext_ape)
            e_pe.append(ext_pe)
            # log_results(file_server + experiment + '/' + cows[i] + '.csv', ext_ape, ext_pe_all, ext_pe)
        tqdm.write('')

    log_all_cows(file_server + experiment + '/all.csv', e_ape, e_pe)

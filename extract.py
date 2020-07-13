import os

import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from data import CLT
from model import segnet

cuda = torch.cuda.is_available()

root_dir = '/data/Databases/s4'
decode = False
batch_size = 32
num_workers = 8
channels = [32, 64, 128, 256, 256]

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

images_dir = os.path.join(root_dir, 'images')
cows = [os.path.join(images_dir, x) for x in os.listdir(images_dir)]
cows = sorted([os.path.basename(x) for x in cows if os.path.isdir(x)])

transform = transforms.Compose([
    transforms.ToTensor()
])

datasets = [CLT(root_dir=root_dir, cows=[cows[i]], decode=False, scale=1, transform=transform) for i in range(len(cows))]
loaders = [data.DataLoader(datasets[i], batch_size=batch_size, shuffle=False, num_workers=num_workers) for i in range(len(datasets))]

model = segnet.SegNet(channels=channels, decode=decode)
model.load_state_dict(torch.load('params_2.8041584491729736_tensor([2.1184, 2.5626, 3.7315])_.pt'))
model = model.to(device)


def iterate(cow, loader):
    model.eval()

    num_samples = 0
    run_err = torch.zeros(3)
    all_err = []

    monitor = tqdm(loader, desc='extract')
    for img, lbl in monitor:
        if decode:
            out, _ = model(img.to(device))
        else:
            out = model(img.to(device))

        num_samples += lbl.size(0)
        err = ((out.detach().cpu() - lbl) ** 2).view(-1, 3, 2).sum(2).sqrt()

        run_err += err.sum(0)
        all_err.append(err)

        monitor.set_postfix(cow=cow, err=(run_err / num_samples).round().tolist(), avg=run_err.mean().item() / num_samples)

    return run_err / num_samples, all_err


def log_results(file_name, err, all_err):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w+')
    file.write('Average Pixel Error for all of the images of this cow,d1,d2,d3\n')
    file.write('{:.5f},'.format(err.mean().item()))
    file.write('{:.5f},'.format(err[0]))
    file.write('{:.5f},'.format(err[1]))
    file.write('{:.5f}'.format(err[2]))
    file.write('\n')
    file.write('Pixel Error values for each individual image (ordered by file names)\n')
    file.write('d1,d2,d3\n')
    for t in all_err:
        t = t.numpy()
        for b in t:
            file.write('{:.5f},'.format(b[0]))
            file.write('{:.5f},'.format(b[1]))
            file.write('{:.5f}'.format(b[2]))
            file.write('\n')
    file.close()


def log_all_cows(file_name, err):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w+')
    file.write('Cow,Average Pixel Error,Pixel Error d1,Pixel Error d2,Pixel Error d3\n')
    for x, y in zip(cows, err):
        file.write(x + ',')
        file.write('{:.5f},'.format(y.mean().item()))
        file.write('{:.5f},'.format(y[0]))
        file.write('{:.5f},'.format(y[1]))
        file.write('{:.5f}'.format(y[2]))
        file.write('\n')
    file.close()


if __name__ == '__main__':

    cows_err = []

    for c, l in zip(cows, loaders):
        with torch.no_grad():
            cow_err, cow_all_err = iterate(c, l)
            cows_err.append(cow_err)
            log_results('results/' + c + '.csv', cow_err, cow_all_err)

    log_all_cows('results/all.csv', cows_err)

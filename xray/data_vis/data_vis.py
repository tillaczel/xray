import os

import matplotlib.pyplot as plt

from xray.utils import load_dataset, XrayDataset


def vis_train_augmentations():
    x, y, idx, n_augmentations = load_dataset('train')
    dataset = XrayDataset(x, y)

    fig = plt.figure(figsize=(24, 12))
    for i in range(32):
        x, y = dataset[i]
        plt.subplot(4, 8, i+1)
        plt.imshow(x[0], cmap='gray')
        plt.title(f'{int(y.numpy()[0]*100)}%')
        plt.xticks([])
        plt.yticks([])
    fig.savefig(os.path.join('results', 'figures', f'augmentations.png'))
    plt.show()


def ordered_xrays():
    x, y, idx, n_augmentations = load_dataset('train')
    dataset = XrayDataset(x, y)

    x_vis = []
    y_vis = []
    for i in range(40):
        x, y = dataset[i*n_augmentations]
        x_vis.append(x.numpy()[0].astype(float))
        y_vis.append(y.numpy()[0].astype(float))

    y_vis, x_vis = zip(*sorted(zip(y_vis, x_vis)))

    fig = plt.figure(figsize=(4, 160))
    for i in range(40):
        plt.subplot(40, 1, i + 1)
        plt.imshow(x_vis[i], cmap='gray')
        plt.title(f'{int(y_vis[i] * 100)}%')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join('results', 'figures', 'ordered.pdf'), bbox_inches='tight')
    plt.close(fig)


def create_dir():
    path = os.path.join('results', 'figures')
    os.makedirs(path, exist_ok=True)

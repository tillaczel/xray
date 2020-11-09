import pandas as pd
import numpy as np
import os
import shutil
import jsonlines
from PIL import Image, ImageOps
import random


def load_dataset(dataset):
    x, y, idx = [], [], []
    labels = pd.read_csv(os.path.join('data', f'target_{dataset}.csv'), index_col=0)
    for im in labels.index.to_numpy():
        try:
            x_i, y_i = get_image(im, dataset, labels)
            x.append(x_i), y.append(y_i), idx.append(str(im))
        except:
            print(f'Skipping {im}')

    list_to_shuffle = list(zip(x, y))
    random.shuffle(list_to_shuffle)
    x, y = zip(*list_to_shuffle)
    y = np.array(y, dtype=float)
    if dataset == 'train':
        y = y[:, np.newaxis]
    return x, y, idx


def get_image(im, dataset, labels):
    path = os.path.join('data', 'raw', f'{im}.png')
    x_i = Image.open(path).convert('L')
    width, height = x_i.size
    if width < height:
        border = int((height - width) / 2)
        x_i = ImageOps.expand(x_i, border=border, fill='black')
        x_i = x_i.crop((0, border, width, height - border), )
    else:
        cut = int((width - height) / 2)
        x_i = x_i.crop((cut, 0, width - cut, height), )

    if dataset == 'train':
        y_i = labels.loc[im].to_numpy()[0]
    elif dataset == 'test':
        y_i = np.reshape(labels.loc[im].to_numpy(), 2)
    return x_i, y_i


def save_dataset(dataset, x, y, idx, n_augmentations=None):
    path = os.path.join('data', 'preprocessed', dataset)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    lables_json = dict()
    for i, im in enumerate(idx):
        x[i].save(os.path.join(path, f'{im}.png'))
        lables_json[im] = list(y[i])

    with jsonlines.open(os.path.join(path, 'db.jsonl'), 'w') as writer:
        writer.write([lables_json, n_augmentations])

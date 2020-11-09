import os

import jsonlines
import numpy as np
import torch
from PIL import Image


def load_dataset(dataset):
    with jsonlines.open(os.path.join('data', 'preprocessed', dataset, 'db.jsonl'), 'r') as reader:
        database_dict, n_augmentations = reader.read()

    x = []
    for idx_i in database_dict:
        x_i = Image.open(os.path.join('data', 'preprocessed', dataset, f'{idx_i}.png'))
        x.append(np.asarray(x_i, dtype=float)[np.newaxis]/255)

    x, y = np.array(x, dtype=float), np.array(list(database_dict.values()), dtype=float)
    idx = np.array(list(database_dict.keys()), dtype=str)
    return x, y, idx, n_augmentations


class XrayDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.n_samples = x.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples
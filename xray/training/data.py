import numpy as np
from torch.utils.data import DataLoader

from xray.utils import load_dataset, XrayDataset


def get_datasets(k_fold, final=False):
    x, y, idx, n_augmentations = load_dataset('train')

    if not final:
        train_datasets = []
        valid_datasets = []
        train_idx = []
        valid_idx = []
        for k in range(k_fold):
            idx_valid = np.arange(int(x.shape[0] * k / k_fold / n_augmentations) * n_augmentations,
                                  int(x.shape[0] * (k + 1) / k_fold / n_augmentations) * n_augmentations)
            idx_train = list(set(range(x.shape[0])) - set(idx_valid))
            train_datasets.append(XrayDataset(x[idx_train], y[idx_train]))
            valid_datasets.append(XrayDataset(x[idx_valid], y[idx_valid]))
            train_idx.append(idx[idx_train])
            valid_idx.append(idx[idx_valid])
    else:
        train_datasets, train_idx = [XrayDataset(x, y)], [idx]
        valid_datasets, valid_idx = None, None

    return train_datasets, valid_datasets, train_idx, valid_idx


def get_loaders(train_datasets, valid_datasets, batch_size, num_workers, final=False):
    train_loaders = [DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) for
                     train_data in train_datasets]
    if not final:
        valid_loaders = [DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) for
                         valid_data in valid_datasets]
    else:
        valid_loaders = None
    return train_loaders, valid_loaders

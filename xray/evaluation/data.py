import torch

from xray.utils import load_dataset, XrayDataset


def get_dataset_and_loader(batch_size, num_workers):
    x, y, idx, n_augmentations = load_dataset(dataset='test')
    test_dataset = XrayDataset(x, y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    return test_dataset, test_loader

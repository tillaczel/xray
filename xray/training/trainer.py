import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from xray.training.data import get_datasets, get_loaders
from xray.training.models import Convnet


class Trainer:

    def __init__(self, k_fold, models, train_datasets, valid_datasets, train_loaders, valid_loaders,
                 criterion, optimizers):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.k_fold = k_fold

        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders

        self.models = models
        self.criterion = criterion
        self.optimizers = optimizers

    def train_step(self, f):
        self.models[f].train()
        train_loss = 0.0
        y_benchmark = 0.0
        for i, (x, y) in enumerate(self.train_loaders[f]):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizers[f].zero_grad()

            outputs = self.models[f](x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizers[f].step()
            train_loss += loss.item() * x.shape[0]
            y_benchmark += np.sum(y.cpu().numpy())
        y_benchmark = y_benchmark / len(self.train_datasets[f])
        return train_loss, y_benchmark

    def valid_step(self, f, y_benchmark):
        self.models[f].eval()
        valid_loss = 0.0
        benchmark_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_loaders[f]):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.models[f](x)
                loss = self.criterion(outputs, y)
                valid_loss += loss.item() * x.shape[0]
                loss = self.criterion(torch.from_numpy(y_benchmark * np.ones(y.shape)).to(self.device), y)
                benchmark_loss += loss.item() * x.shape[0]
        return valid_loss, benchmark_loss

    def train(self, n_epochs, final=False):
        loss_history = np.zeros((self.k_fold, n_epochs, 3))
        for f in range(self.k_fold):
            self.models[f].to(self.device)
            if not final:
                desc = f'Cross validation {f+1}/{self.k_fold}'
            else:
                desc = 'Final model training'
            for epoch in tqdm(range(n_epochs), desc=desc):
                train_loss, y_benchmark = self.train_step(f)
                train_loss = train_loss / len(self.train_datasets[f])

                if not final:
                    valid_loss, benchmark_loss = self.valid_step(f, y_benchmark)
                    valid_loss = valid_loss / len(self.valid_datasets[f])
                    benchmark_loss = benchmark_loss / len(self.valid_datasets[f])
                else:
                    valid_loss, benchmark_loss = None, None

                loss_history[f, epoch] = [train_loss, valid_loss, benchmark_loss]
            self.vis_training(loss_history, final=final, f=f)
            self.models[f].cpu()
        if not final:
            self.vis_training(loss_history, final=final)
        return self.models, loss_history

    @staticmethod
    def vis_training(loss_history, f=None, final=False):
        if f is None:
            y = np.mean(loss_history, axis=0)
        else:
            y = loss_history[f]
        loss_mean_percentage = np.round(np.mean(y[-10:], axis=0)*100, 4)

        fig = plt.figure(figsize=(12, 8))
        plt.plot(y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if not final:
            plt.ylim(-y[-1, -1] * 0.1, y[-1, -1] * 1.1)
            plt.legend([f'Train loss: {loss_mean_percentage[0]}%', f'Validation loss: {loss_mean_percentage[1]}%',
                        f'Benchmark loss: {loss_mean_percentage[2]}%'])
        else:
            plt.ylim(-np.mean(y[:, 0]) * 0.1, np.mean(y[:, 0]) * 1.1)
            plt.legend([f'Train loss: {loss_mean_percentage[0]}%'])

        if f is None:
            title = 'Average loss over epochs'
        elif final:
            title = 'Final loss over epochs'
        else:
            title = f'Loss of model {f} over epochs'
        plt.title(title)

        fig.savefig(os.path.join('results', 'figures', 'loss', f'{title}.png'))
        plt.show()


def training_process(k_fold, batch_size, n_epochs, num_workers, learning_rate, final=False):
    train_datasets, valid_datasets, train_idx, test_idx = get_datasets(k_fold, final=final)

    train_loaders, valid_loaders = get_loaders(train_datasets, valid_datasets, batch_size, num_workers, final=final)

    models = [Convnet() for _ in range(k_fold)]
    criterion = nn.L1Loss()
    optimizers = [optim.Adam(models[f].parameters(), lr=learning_rate) for f in range(k_fold)]

    trainer = Trainer(k_fold, models, train_datasets, valid_datasets, train_loaders, valid_loaders,
                      criterion, optimizers)
    models, loss_history = trainer.train(n_epochs, final=final)

    if not final:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), os.path.join('results', 'models', f'train_{i}.pt'))
        pd.DataFrame(np.mean(loss_history, axis=0)).to_csv(os.path.join('results', 'loss_history_kfold.csv'))
    else:
        torch.save(models[0].state_dict(), os.path.join('results', 'models', 'final.pt'))
        pd.DataFrame(np.mean(loss_history, axis=0)).to_csv(os.path.join('results', 'loss_history_final.csv'))


def delete_previous_results():
    path = os.path.join('results', 'figures', 'loss')
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    path = os.path.join('results', 'models')
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)




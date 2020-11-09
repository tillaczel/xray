import torch
import torch.nn as nn

import os
import numpy as np
import jsonlines

from xray.training.models import Convnet


def get_loss_percentage(dataset, loader, y_pred, criterion):
    loss_cum_expert = 0.0
    loss_cum_benchmark = 0.0
    for i, (X, Y) in enumerate(loader):
        loss = criterion(torch.from_numpy(y_pred * np.ones(Y[:, 0:1].shape)), Y[:, 0:1])
        loss_cum_benchmark += loss.item() * X.shape[0]
        loss = criterion(Y[:, 1:2], Y[:, 0:1])
        loss_cum_expert += loss.item() * X.shape[0]
    loss_cum_benchmark = np.round(loss_cum_benchmark / len(dataset) * 100, 4)
    loss_cum_expert = np.round(loss_cum_expert / len(dataset) * 100, 4)
    return loss_cum_expert, loss_cum_benchmark


def get_model_loss_percentage(dataset, loader, models, criterion):
    loss_cum = 0.0
    with torch.no_grad():
        [model.eval() for model in models]
        for i, (X, Y) in enumerate(loader):
            y_pred = torch.mean(torch.Tensor([model(X).numpy() for model in models]), dim=0)
            loss = criterion(y_pred, Y[:, 0:1])
            loss_cum += loss.item() * X.shape[0]
    loss_cum = np.round(loss_cum / len(dataset) * 100, 4)
    return loss_cum


def load_model(model_name):
    model = Convnet()
    model.load_state_dict(torch.load(os.path.join('results', 'models', f'{model_name}.pt')))
    return model


def evaluate_model_losses(test_dataset, test_loader, criterion, k_fold):
    with jsonlines.open(os.path.join('data', 'preprocessed', 'train', 'db.jsonl'), 'r') as reader:
        database_dict, _ = reader.read()
    y_benchmark = np.mean(list(database_dict.values()))
    expert_loss, benchmark_loss = get_loss_percentage(test_dataset, test_loader, y_benchmark, criterion)
    print(f'Expert loss: {expert_loss}%')
    print(f'Benchmark loss: {benchmark_loss}%')

    final_loss = get_model_loss_percentage(test_dataset, test_loader, [load_model('final')], criterion)
    print(f'Final loss: {final_loss}%')

    models = [load_model(f'train_{i}') for i in range(k_fold)]
    ensemble_loss = get_model_loss_percentage(test_dataset, test_loader, models, criterion)
    print(f'Ensemble loss: {ensemble_loss}%')

    for i in range(k_fold):
        loss_i = get_model_loss_percentage(test_dataset, test_loader, [models[i]], criterion)
        print(f'Model {i} loss: {loss_i}%')

from torch import nn

from xray.data_vis.data_vis import create_dir, vis_train_augmentations, ordered_xrays

from xray.preprocessing.augment import augment
from xray.preprocessing.augment import resize
from xray.preprocessing.loading_and_saving import load_dataset, save_dataset

from xray.training.trainer import training_process, delete_previous_results
from xray.training.model_architecture import get_summary

from xray.evaluation.data import get_dataset_and_loader
from xray.evaluation.utils import evaluate_model_losses


class Experiment:

    def __init__(self):
        pass

    def preprocess(self):
        x_train, y_train, idx_train = load_dataset(dataset='train')
        x_test, y_test, idx_test = load_dataset(dataset='test')

        x_train, y_train, idx_train, n_augmentations = augment(x_train, y_train, idx_train)
        x_test = resize(x_test)

        save_dataset('train', x_train, y_train, idx_train, n_augmentations)
        save_dataset('test', x_test, y_test, idx_test)

    def vis_data(self):
        create_dir()
        vis_train_augmentations()
        ordered_xrays()

    def train(self, k_fold, batch_size, n_epochs, num_workers, learning_rate):
        get_summary()
        delete_previous_results()

        # K fold cross validation
        training_process(k_fold, batch_size, n_epochs, num_workers, learning_rate, final=False)

        # Train final model
        training_process(1, batch_size, n_epochs, num_workers, learning_rate, final=True)

    def evaluate(self, batch_size, num_workers, k_fold):
        test_dataset, test_loader = get_dataset_and_loader(batch_size, num_workers)
        evaluate_model_losses(test_dataset, test_loader, nn.L1Loss(), k_fold)




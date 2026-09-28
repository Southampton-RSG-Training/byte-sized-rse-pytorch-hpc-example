from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from simple_cnn.model import SimpleCNN, train_epoch, test, load, save
from simple_cnn.data import download_mnist_data, load_mnist_data
from simple_cnn.utils import quiet


class TestSimpleCNN(TestCase):

    def load_data(self, data_directory, train: bool = True):
        mnist = load_mnist_data(data_directory, None, train)
        yield mnist

    def test_train_test(self):
        """Basic testing of running the SimpleCNN model"""
        model = SimpleCNN()
        device = torch.device("cpu")
        model.to(device)
        optimizer = Adadelta(model.parameters())

        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir) / "data"
            checkpoint_directory = Path(tmpdir) / "checkpoints"
            with quiet():
                download_mnist_data(data_directory)

                training_data = load_mnist_data(data_directory, None)
                training_loader = DataLoader(training_data, batch_size=64)

                train_epoch(
                    model, device, training_loader, optimizer, epoch=1, dry_run=True
                )

                test_data = load_mnist_data(data_directory, None)
                test_loader = DataLoader(test_data, batch_size=32)

                test(model, device, test_loader)

                save(model, 1, checkpoint_directory)
                epoch = load(
                    model, sorted(checkpoint_directory.glob("SimpleCNN_*.pt"))[-1]
                )

                self.assertEqual(epoch, 1)

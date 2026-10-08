from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from accelerate import Accelerator, logging
import torch
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from simple_cnn.model import SimpleCNN, train_epoch, test, load, save, train
from simple_cnn.data import download_mnist_data, load_mnist_data
from simple_cnn.utils import quiet


class TestSimpleCNN(TestCase):

    def load_data(self, data_directory, train: bool = True):
        mnist = load_mnist_data(data_directory, None, train)
        yield mnist

    def test_train_epoch_and_test(self):
        """Basic testing of training and running the SimpleCNN model"""
        accelerator = Accelerator()

        model = SimpleCNN()
        optimizer = Adadelta(model.parameters())

        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir) / "data"
            checkpoint_directory = Path(tmpdir) / "checkpoints"
            with quiet():
                download_mnist_data(data_directory)

                training_data = load_mnist_data(data_directory, None)
                training_loader = DataLoader(training_data, batch_size=64)

                test_data = load_mnist_data(data_directory, None)
                test_loader = DataLoader(test_data, batch_size=32)

                model, optimizer, training_loader, test_loader = accelerator.prepare(
                    model, optimizer, training_loader, test_loader
                )

                train_epoch(
                    model, accelerator, training_loader, optimizer, epoch=1,
                )

                test(model, accelerator, test_loader)

                save(accelerator, 1, checkpoint_directory)
                checkpoints = sorted(checkpoint_directory.glob("SimpleCNN_*"))
                self.assertNotEqual(checkpoints, [])
                epoch = load(accelerator, checkpoints[-1])

                self.assertEqual(epoch, 1)

    def test_train(self):
        model = SimpleCNN()
        optimizer = Adadelta(model.parameters())
        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir) / "data"
            checkpoint_directory = Path(tmpdir) / "checkpoints"
            with quiet():
                download_mnist_data(data_directory)

                training_data = load_mnist_data(data_directory, None)
                training_loader = DataLoader(training_data, batch_size=64)

                test_data = load_mnist_data(data_directory, None)
                test_loader = DataLoader(test_data, batch_size=32)

                train(model, optimizer, training_loader, test_loader, 2, checkpoint_directory)

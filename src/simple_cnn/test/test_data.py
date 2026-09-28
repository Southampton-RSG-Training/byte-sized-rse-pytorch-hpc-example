from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from simple_cnn.data import download_mnist_data, load_mnist_data, MNIST
from simple_cnn.utils import quiet


class TestData(TestCase):

    def test_download(self):
        """Test downloading the MNIST data"""
        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir)

            with quiet():
                download_mnist_data(data_directory)

            self.assertListEqual(
                list(data_directory.iterdir()), [data_directory / "MNIST"]
            )

    def test_load_train(self):
        """Test loading the MNIST training data"""
        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir)
            with quiet():
                download_mnist_data(data_directory)

                mnist = load_mnist_data(data_directory, train=True)

            self.assertIsInstance(mnist, MNIST)
            self.assertEqual(len(mnist), 60_000)

    def test_load_test(self):
        """Test loading the MNIST test data"""
        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir)
            with quiet():
                download_mnist_data(data_directory)

                mnist = load_mnist_data(data_directory, train=False)

            self.assertIsInstance(mnist, MNIST)
            self.assertEqual(len(mnist), 10_000)

    def test_load_missing(self):
        """Test error raised when trying to load missing MNIST data"""
        with TemporaryDirectory() as tmpdir:
            data_directory = Path(tmpdir)
            with self.assertRaises(RuntimeError):
                load_mnist_data(data_directory)

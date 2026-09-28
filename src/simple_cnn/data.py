from pathlib import Path
from typing import Any, Callable, ContextManager

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from .utils import quiet


def download_mnist_data(
    data_directory: Path, redirect: Callable[[], ContextManager[Any]] = quiet
):
    """Download the MNIST data into a known location.

    This should be run before running on the IridisX cluster.

    Parameters
    ----------
    data_directory : Path
        The Path that the data should be downloaded to.
    """
    data_directory.mkdir(parents=True, exist_ok=True)
    MNIST(data_directory, download=True)


def load_mnist_data(
    data_directory: Path,
    transform: Callable[[Any], Any] | None = None,
    train: bool = True,
    redirect: Callable[[], ContextManager[Any]] = quiet,
) -> MNIST:
    """Load the MNIST data from a know location.

    This loads from already downloaded data.

    Parameters
    ----------
    data_directory : Path
        The Path that the data should br loaded from.
    transform : Callable | None
        A callable that takes a PIL image and returns a PIL image, or None.
    train : bool
        Whether to load the training or test data.

    Returns
    -------
    mnist : torch.dataset.Dataset
        The dataset containing the MNIST images
    """
    if transform is None:
        # default normalization
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )
    return MNIST(data_directory, transform=transform, train=train, download=False)

import logging
from pathlib import Path
import os

import click


@click.group()
@click.option(
    "--log-dir",
    "log_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="",
)
def cli(log_dir_name):
    if log_dir_name:
        # ensure logging directory exists
        log_dir = Path(log_dir_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"log-{os.getpid()}.log",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option(
    "--data-dir",
    "data_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="data",
)
def preprocess(data_dir_name: str):
    """Perform preprocessing commands before distributing to nodes."""
    import evaluate
    from .data import download_mnist_data

    # ensure data directory exists
    data_dir = Path(data_dir_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    download_mnist_data(data_dir)

    # install metrics from HuggingFace now
    evaluate.load("accuracy")


@cli.command()
@click.option(
    "--data-dir",
    "data_dir_name",
    type=click.Path(exists=True, file_okay=False),
    default="data",
)
@click.option(
    "--checkpoints-dir",
    "checkpoints_dir_name",
    type=click.Path(file_okay=False, writable=True),
    default="checkpoints",
)
@click.option("--split", default=0.8)
@click.option("--epochs", default=10)
@click.option("--resume/--no-resume")
def train(
    data_dir_name: str,
    checkpoints_dir_name: str,
    split: float,
    epochs: int,
    resume: bool,
):
    """Train on the downloaded data."""
    try:
        from torch.optim import Adadelta
        from torch.utils.data import DataLoader, random_split

        from .model import SimpleCNN, train
        from .data import load_mnist_data

        data_dir = Path(data_dir_name)
        checkpoints = Path(checkpoints_dir_name)
        if resume:
            start = None
        else:
            start = 1
        # ensure checkpoints directory exists
        checkpoints.mkdir(parents=True, exist_ok=True)


        model = SimpleCNN()
        optimizer = Adadelta(model.parameters())

        dataset = load_mnist_data(data_dir)
        training_data, validation_data = random_split(dataset, [split, 1 - split])
        training_loader = DataLoader(training_data, batch_size=64)
        validation_loader = DataLoader(validation_data, batch_size=64)

        train(
            model,
            optimizer,
            training_loader,
            validation_loader,
            epochs,
            checkpoints,
            start=start,
        )
    except Exception:
        from accelerate import logging

        logger = logging.get_logger(__name__)
        logger.exception("Exception during training:")

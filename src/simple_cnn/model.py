import datetime
import io
import pickle
import re
from math import inf

from accelerate import Accelerator, logging
import torch
from torch import flatten, no_grad
from torch.nn import Module, Conv2d, Dropout, Linear
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split

logger = logging.get_logger(__name__)


class SimpleCNN(Module):
    """CNN for MNIST Data

    From PyTorch examples: https://github.com/pytorch/examples/blob/main/mnist/main.py
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(9216, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output


def train_epoch(
    model,
    accelerator,
    train_loader,
    optimizer,
    epoch,
    log_interval=10,
    dry_run=False,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nll_loss(output, target)
        accelerator.backward(loss)
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break


def predict(model, data):
    model.eval()
    with no_grad():
        # data = data.to(device)
        return model(data)


def test(model, test_loader):
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            output = predict(model, data)
            # target = target.to(device)
            test_loss += nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss


def train(
    model,
    optimizer,
    train_loader,
    validation_loader,
    epochs,
    checkpoint_dir,
    start=1,
    patience=None,
    gamma=0.7,
    log_interval=10,
    dry_run=False,
):
    best_val_loss = inf
    if patience is None:
        patience = epochs
    wait = 0
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    accelerator = Accelerator()

    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )

    if start is None:
        # try to resume if there are any checkpoints
        checkpoints = sorted(checkpoint_dir.glob("SimpleCNN_*"))
        if checkpoints:
            start = load(accelerator, checkpoints[-1]) + 1
        else:
            start = 1

    for epoch in range(start, start + epochs):
        train_epoch(
            model, accelerator, train_loader, optimizer, epoch, log_interval, dry_run
        )
        val_loss = test(model, validation_loader)
        scheduler.step()
        if val_loss < best_val_loss:
            # checkpoint if model is better than previous best
            save(accelerator, epoch, checkpoint_dir)
            wait = 0
        else:
            wait += 1
            if wait > patience:
                logger.info(f"Training stabilised at epoch {epoch}.")
                break


def save(accelerator, epoch, path):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"SimpleCNN_{epoch:0>5d}"
    accelerator.save_state(path / filename)
    logger.info(f"Checkpoint saved: {path}/{filename}")


def load(accelerator, path):
    accelerator.load_state(path)
    epoch = int(path.name.split("_")[-1])
    return epoch

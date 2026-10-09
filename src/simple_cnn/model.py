from math import inf

from accelerate import Accelerator, logging
import evaluate
from torch import flatten, no_grad
from torch.nn import Module, Conv2d, Dropout, Linear
from torch.nn.functional import relu, max_pool2d, log_softmax, nll_loss
from torch.optim.lr_scheduler import StepLR

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
                "Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * len(data),
                    batch_idx / len(train_loader),
                    loss.item(),
                ),
                main_process_only=False,
            )
            if dry_run:
                break


def predict(model, data):
    model.eval()
    with no_grad():
        # data = data.to(device)
        return model(data)


def test(model, accelerator, test_loader):
    metric = evaluate.load("accuracy")
    with no_grad():
        for data, target in test_loader:
            output = predict(model, data)
            predictions = output.argmax(dim=-1)
            all_predictions, all_targets = accelerator.gather_for_metrics(
                (predictions, target)
            )
            metric.add_batch(
                predictions=all_predictions,
                references=all_targets,
            )

    accuracy = metric.compute()['accuracy']
    logger.info(
        "Test set: Accuracy: {}\n".format(accuracy)
    )
    return accuracy


def train(
    model,
    optimizer,
    train_loader,
    validation_loader,
    epochs,
    checkpoint_dir,
    start=1,
    gamma=0.7,
    log_interval=10,
    dry_run=False,
):
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
        test(model, accelerator, validation_loader)
        scheduler.step()

        # checkpoint
        save(accelerator, epoch, checkpoint_dir)


def save(accelerator, epoch, path):
    path.mkdir(parents=True, exist_ok=True)
    filename = f"SimpleCNN_{epoch:0>5d}"
    accelerator.save_state(path / filename)
    logger.info(f"Checkpoint saved: {path}/{filename}")


def load(accelerator, path):
    accelerator.load_state(path)
    epoch = int(path.name.split("_")[-1])
    return epoch

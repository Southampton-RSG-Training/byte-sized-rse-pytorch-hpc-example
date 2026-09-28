SimpleCNN Example
=================

This is an example of a SimpleCNN model with the ability to checkpoint and
resume.  It's intended to be a good exmaple of how to write a model that can
be trained on IridisX.

This uses the MNIST digits dataset as the training data.

Installation
------------

Starting with a working Python 3.13 or 3.14, create a virtual environment and
activate it:

..  code-block: console

    python3.14 -m venv env
    source env/bin/activate

Then install the project with the dev options:

..  code-block: console

    pip install -e ".[dev]"


Testing
-------

You can test your install using unitest discover:

..  code-block: console

    python -m unittest discover --s src -v

This runs the tests using the CPU only.

You can verify that the cli has been installed by running:

..  code-block: console

    simple-cnn --help

Usage
-----

On the login node run the preprocessing script to download the MNIST data:

..  code-block: console

    simple-cnn preprocess

By default the data is downloaded to a ``data`` directory in the current
working directory.  This can be overidden with the ``--data-dir`` option.

Local Training
~~~~~~~~~~~~~~

To train the model on the downloaded data use:

..  code-block: console

    simple-cnn train

By default it performs 10 epochs of training with a 80%/20% train/validation
split on the CPU, saving checkpoints as long as the validation loss is an
improvement.  These defaults can be changed with command-line arguments.
For example to train for 5 epochs on an Apple Silicon Mac's GPU, you would
use:

..  code-block: console

    simple-cnn train --device=mps --epochs=5

By default the training starts from scratch each time. To resume from the
last checkpoint, use the ``--resume`` argument. For example to do ten more
epochs of training you would call:

..  code-block: console

    simple-cnn train --device=mps --epochs=10 --resume

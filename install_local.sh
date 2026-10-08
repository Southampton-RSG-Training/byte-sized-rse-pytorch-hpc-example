#!/usr/bin/env bash

# Bash script to install python environment and download data

WORKING_DIRECTORY=.
VENV_NAME=$WORKING_DIRECTORY/venv
DATA_DIR=$WORKING_DIRECTORY/data

# create virtual environment
python3.14 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# install source and dependencies
pip install -U pip
pip install -e .

# run preprocessing script
simple-cnn preprocess --data-dir $DATA_DIR

# deactivate virtual environment
deactivate

#!/bin/bash

#SBATCH --job-name=simple-cnn-example
#SBATCH --partition=a100
#SBATCH --time=00:05:00
#SBATCH --nodes=1                         # Number of Nodes (max=1)
#SBATCH --gpus=1                          # GPUs per Node (max=8)
#SBATCH --ntasks=1                        # Number of Nodes x GPUs per Node
#SBATCH --gpus-per-task=1                 # Every task to use two GPUs
#SBATCH --cpus-per-task=1                 # Every task to use two CPUs
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding

WORKING_DIRECTORY=simple_cnn_workspace
VENV_NAME=$WORKING_DIRECTORY/venv
DATA_DIR=$WORKING_DIRECTORY/data
CHECKPOINTS_DIR=$WORKING_DIRECTORY/checkpoints
LOG_DIR=$WORKING_DIRECTORY/logs

# Optional: print useful job info
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS_PER_TASK"

# Load required modules
module purge
module load python
module load cuda

# Activate Python virtual environment
source $VENV_NAME/bin/activate

# Set any environment variables or configuration options
export PYTHONUNBUFFERED=1                  # Use unbuffered stdout/stderr

# Move to job directory
cd $SLURM_SUBMIT_DIR

# Run the Python script
accelerate launch --num_processes 1 --no_python simple-cnn train --data-dir $DATA_DIR --checkpoints-dir $CHECKPOINTS_DIR --epochs=10

# deactivate virtual environment
deactivate

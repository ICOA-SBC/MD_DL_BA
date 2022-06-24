#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=videonucy_training
#SBATCH --account=dzc@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.11.0

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python videonucy_train.py training.learning_rate=1e-4 \
            training.weight_decay=0 \
            data_setup.frames=30 \
            training.batch_size=32 \
            training.num_epochs=150 \
            training.patience=30 

date
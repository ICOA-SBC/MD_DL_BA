#!/bin/bash
#SBATCH --job-name=videonucy_training
#SBATCH --account=sos@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g

module purge
module load pytorch-gpu/py3/1.11.0

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python videonucy_train.py \
            training.learning_rate=1e-5 \
            training.weight_decay=1e-4 \
            data_setup.frames=50 \
            training.batch_size=8 \
            training.num_epochs=100 \
            training.patience=30 

date
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sysrot
#SBATCH --account=<tellme>
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.10.0

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python proli_systematicrotations.py training.num_epochs=50 name=rotations \
                  training.learning_rate=0.0001 training.weight_decay: 0.0005

date

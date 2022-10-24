#!/bin/bash
#SBATCH --job-name=densenucy_training_multi-gpu
#SBATCH --output=log/%x%j.out
#SBATCH --error=log/%x%j.out
#SBATCH --account=dzc@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.11.0

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python proli_dense_multi-gpu.py

date

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=densenucy_training
#SBATCH --account=dzc@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
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

time srun -u python proli-dense.py

date

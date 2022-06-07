#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=test
#SBATCH --account=sos@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --time=06:00:00
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

time srun -u python proli.py training.num_epochs=2 experiment_name=test_cp mlflow.run_name=simple

date

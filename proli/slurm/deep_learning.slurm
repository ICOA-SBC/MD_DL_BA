#!/bin/bash
#SBATCH --account=dzc@v100
#SBATCH --job-name=test
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err
#SBATCH --time=1:00:00
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.10.0

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python proli_systematic_rotations.py \
    io.input_dir=$dzc_ALL_CCFRWORK/deep_learning/pafnucy/data/subset_no_structure_all_frame \
    mlflow.run_name=run_cp \
    experiment_name=expr_cp

date

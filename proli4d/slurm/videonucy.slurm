#!/bin/bash
#SBATCH --job-name=videonucy_training
#SBATCH --account=luh@v100
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
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python videonucy_train.py \
            experiment.name=videonucy_v3 \
            experiment.run=training1 \
            experiment.by_complex=false \
            training.learning_rate=1e-4 \
            training.weight_decay=1e-7 \
            data_setup.frames=50 \
            training.batch_size=12 \
            training.num_epochs=100 \
            training.patience=20

date
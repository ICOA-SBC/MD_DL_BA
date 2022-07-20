#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=dsv2_by_sim_lrplateau
#SBATCH --account=sos@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python main_train.py training.learning_rate=1e-2 \
            training.weight_decay=1e-3 \
            data_setup.frames=40 \
            training.batch_size=16 \
            training.num_epochs=250 \
            training.patience=50 \
            experiment.name=datasetv2 \
            experiment.run=adam_lrp_ds2_bysim \
            experiment.by_complex=false

date

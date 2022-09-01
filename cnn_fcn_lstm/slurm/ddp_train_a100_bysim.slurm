#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=long_sim
#SBATCH --account=sos@a100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --constraint a100
#SBATCH --hint=nomultithread

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date
time srun -u python -u main_train.py training.learning_rate=1e-3 \
            training.weight_decay=1e-5 \
            data_setup.frames=40 \
            training.batch_size=32 \
            training.num_epochs=300 \
            training.patience=40 \
            training.warmup_epochs=30 \
            experiment.run=long_sim_4a100 \
            experiment.name=ddp_a100 \
            experiment.by_complex=false \
            model.architecture=single_densecnn \
            debug=false
            
echo " "
echo "_________________________________________________________________ "
echo "RUNNING Test only
echo "_________________________________________________________________"
echo " "

time srun -u python -u main_test.py training.learning_rate=1e-2 \
            experiment.name=ddp_test \
            experiment.by_complex=false \
            experiment.run=save_load \
            training.batch_size=32 \
            data_setup.frames=40 \
            model.architecture=single_densecnn \
            debug=false

date

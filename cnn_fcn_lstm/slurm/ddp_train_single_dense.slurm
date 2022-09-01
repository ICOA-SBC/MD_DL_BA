#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=train_4gpu
#SBATCH --account=sos@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

# avoids warning from mlflow
export GIT_PYTHON_REFRESH=quiet

set -x

date

time srun -u python -u main_train.py training.learning_rate=1e-3 \
            training.weight_decay=1e-5 \
            data_setup.frames=10 \
            training.batch_size=16 \
            training.num_epochs=10 \
            training.patience=25 \
            training.warmup_epochs=40 \
            experiment.run=test_simple \
            experiment.name=ddp_test \
            model.architecture=single_densecnn \
            debug=false

echo " "
echo "_________________________________________________________________ "
echo "RUNNING SECOND PART: TEST !!(which doesn't implement ddp)"
echo "_________________________________________________________________"
echo " "

time srun -u python -u main_test.py \
            experiment.name=ddp_test \
            experiment.run=test_simple \
            data_setup.frames=10 \
            model.architecture=single_densecnn \
            debug=false

date

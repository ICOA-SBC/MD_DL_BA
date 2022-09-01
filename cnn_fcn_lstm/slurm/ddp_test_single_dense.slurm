#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=test_1gpu
#SBATCH --account=sos@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
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

echo " "
echo "_________________________________________________________________ "
echo "RUNNING Test only"
echo "_________________________________________________________________"
echo " "

time srun -u python -u main_test.py \
            experiment.name=ddp_test \
            experiment.run=test_simple \
            data_setup.frames=10 \
            model.architecture=single_densecnn 
date

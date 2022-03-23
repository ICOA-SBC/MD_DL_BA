# Environnement setup

## Running the original pafnucy code

* [Git](https://gitlab.com/cheminfIBB/pafnucy)
* `environment_gpu.yml` comes from the git repository

### Creation of the conda environment

The goal is to share the same environment between users:

```
export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_tf1.2
export CONDA_ENVS_PATH=$luh_ALL_CCFRWORK/conda
mkdir $PYTHONUSERBASE $CONDA_ENVS_PATH

module purge
module load anaconda-py3/2021.05
conda env create -f environment_gpu.yml --prefix $luh_ALL_CCFRWORK/conda/tf1.2
conda activate tf1.2
```

If we ever need to add some libraries:

```
export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_tf1.2
export CONDA_ENVS_PATH=$luh_ALL_CCFRWORK/conda
module load anaconda-py3/2021.05
conda activate tf1.2
```

then, either:

```
conda install <package>
```

or 

```
pip install --no-cache-dir --user <package>
```

### Usage 


```
module purge
export CONDA_ENVS_PATH=$luh_ALL_CCFRWORK/conda
module load anaconda-py3/2021.05

export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_tf1.2

conda activate tf1.2
```

## Running pytorch version of pafnucy within a module (WIP)

**Note**: Versions of libraries are not the same as in the original pafnucy

### Update to module pytorch 1.10.1

```commandline
module load pytorch-gpu/py3/1.10.1
export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_pt10.1
pip install --no-cache-dir --user openbabel
```
> Fail ... (needs to compile openbabel, then install python wrapper)

### Usage

```commandline
module load pytorch-gpu/py3/1.10.1
export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_pt10.1
export PATH=$luh_ALL_CCFRWORK/.local_pt10.1/bin:$PATH
```

## Running pytorch version of pafnucy within a conda environment (temporary solution, WIP)

### Creation

Changes in version :
* openbabel-3.1.1 (instead originally of 2.4.1)

```commandline
srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --hint=nomultithread --time=02:00:00 \
    --partition=prepost --account xxx bash

export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_pt1.11
export CONDA_ENVS_PATH=$luh_ALL_CCFRWORK/conda
mkdir $PYTHONUSERBASE $CONDA_ENVS_PATH

module purge
module load anaconda-py3/2021.05
conda create -n pt1.11 python=3.9
conda activate pt1.11

conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install openbabel -c openbabel
conda install h5py -c conda-forge
```

> this solution is not good when looking for performance
> 
> see https://github.com/openbabel/openbabel/tree/master/scripts/python
> 
> `conda remove --name pt1.11 --all` to delete environment

### Usage

```commandline
export PYTHONUSERBASE=$luh_ALL_CCFRWORK/.local_pt1.11
export CONDA_ENVS_PATH=$luh_ALL_CCFRWORK/conda
export PATH=$luh_ALL_CCFRWORK/.local_pt1.11/bin:$PATH

module purge
module load anaconda-py3/2021.05
conda activate pt1.11
```
> fail : `import pybel` is not working 

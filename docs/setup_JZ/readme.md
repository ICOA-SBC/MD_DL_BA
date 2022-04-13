# Environment setup

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



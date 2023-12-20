# Proli: a pytorch version of Pafnucy

## Setup

* [hydra-core](https://hydra.cc/docs/intro/) is used for setting up all the options
* [mlflow](https://mlflow.org/) is used to build an experiment notebook

```commandline
conda env create -f environment.yml
conda activate MD_DL_BA_env
pip install --user --no-cache-dir hydra-core mlflow
```

## Usage

Command line:

```commandline
python proli.py training.num_epochs=100 mlflow.runname=<name>
```

This will take all the parameters setup from hydra (see `configs/proli.yaml`) and will only update the number of epochs.

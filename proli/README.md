# proli: a pytorch version of Pafnucy

proli works with [hydra](https://hydra.cc/docs/intro/) and [mlflow](https://mlflow.org/) for experiment setup and logging.  

## Setup

This code required several additional libraries: 

```commandline
module load pytorch-gpu/py3/1.10.0
pip install --no-cache-dir --user mlflow
pip install --no-cache-dir --user hydra-core
pip install --no-cache-dir --user torchinfo
```

## Usage

Command line:

```commandline
python proli.py training.num_epochs=100 mlflow.runname=<name>
```

This will take all the parameters setup from hydra (see `configs/default.yaml`) and will only update the number of epochs.


See slurm file for an example of a job (don't forget to update your account!).

If you keep the default for mlrun, the experiment logs will be in `$luh_ALL_CCFRSCRATCH/proli/mlruns`.

## Todos

Many things, in particular:

* **saving, loading models**
* more metrics on test dataset
* ...
# Readme

## Setup

```
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

pip install --no-cache-dir --user mlflow==1.27.0 
```

Which install:
```
Installing collected packages: sqlparse, querystring-parser, pyjwt, Mako, gunicorn, greenlet, sqlalchemy, docker, databricks-cli, prometheus-flask-exporter, alembic, mlflow
Successfully installed Mako-1.2.1 alembic-1.8.0 databricks-cli-0.17.0 docker-5.0.3 greenlet-1.1.2 gunicorn-20.1.0 mlflow-1.27.0 prometheus-flask-exporter-0.20.2 pyjwt-2.4.0 querystring-parser-1.2.4 sqlalchemy-1.4.39 sqlparse-0.4.2
```

> Attention les versions mlflow évoluent vite, d'où l'intérêt de figer une version ! 

## Usage

In a slurm file:

```
module load cpuarch/amd #add this line only if using A100
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH
```

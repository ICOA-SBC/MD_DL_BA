# MD_DL_BA

Implementation of deep neural networks to predict the binding affinity of protein-ligand complexes from molecular dynamic simulations.

Proli and Densenucy, both can be trained with MD data augmentation.
Timenucy and Videonucy are spatio-temporal learning methods that use 4D data (entire molecular dynamics simulations).

## Setup

* hydra-core is used for setting up all the options
* mlflow is used to build an experiment notebook

One environment is used for all the neural networks.

``` 
conda install environment.yml
conda activate MD_DL_BA_env
pip install --user --no-cache-dir hydra-core mlflow
```


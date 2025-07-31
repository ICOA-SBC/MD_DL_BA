# MD_DL_BA

Code for the paper: "Spatio-temporal learning from molecular dynamics simulations for protein-ligand binding affinity prediction" by Pierre-Yves Libouban, Camille Parisel, Maxime Song, Samia Aci-Sèche, Jose C. Gómez-Tamayo, Gary Tresadern, and Pascal Bonnet

Implementation of 4 deep neural networks - Proli, Densenucy, Timenucy and Videonucy - to predict the binding affinity of protein-ligand complexes from molecular dynamic simulations.

Both, Proli and Densenucy, can be trained with MD data augmentation.
Timenucy and Videonucy are spatio-temporal learning methods that use 4D data (entire molecular dynamics simulations).

## Setup

* hydra-core is used for setting up all the options
* mlflow is used to build an experiment notebook

One environment is used for all the neural networks.

``` 
conda install environment.yml
conda activate MD_DL_BA_env
pip install --user --no-cache-dir hydra-core mlflow
conda install -c conda-forge unzip
```

## Data
Training/validation and test datasets can be downloaded from zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10390550.svg)](https://doi.org/10.5281/zenodo.10390550).
To download the training/validation and test data, excute the following code:
``` 
conda activate MD_DL_BA_env
cd ./datasets
bash download_train_data.sh
bash download_test_data.sh
```

Raw Data (Molecular dynamics simulation) without any postprocessing can be visualized and downloaded (4.5 To) on the [MDDB](https://mdposit.mddbr.eu/#/browse?search=MDBind)

## Training/testing
The workflow is described in the readme files for each neural network

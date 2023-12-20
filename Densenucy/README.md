# Densenucy an improved version of Pafnucy/Proli based on Densenet architecture

## Setup

* [hydra-core](https://hydra.cc/docs/intro/) is used for setting up all the options
* [mlflow](https://mlflow.org/) is used to build an experiment notebook

```commandline
conda install environment.yml
conda activate MD_DL_BA_env
```

## Training

Densenucy is used with random rotations to train with MD data augmentation in a reasonable amount of time with `densenucy.py`.
Nonetheless, it is also possible to train densenucy with systematic rotations (24 rotations) using `densenucy_systematic_rotations.py`.

Training is setup to work with multi-gpu (or nodes).
When using MD data augmentation, loading the training set takes a long time.

The training/validation set need to be downloaded from zenodo.


```commandline
conda activate MD_DL_BA_env
python densenucy.py
```

```commandline
conda activate MD_DL_BA_env
python densenucy_systematic_rotations.py
```

## Testing

Metrics are computed at the end of the training in `densenucy.py` but it is possible to evaluate models on a test with `densenucy.py` by using `training.only_test`.

It is required to know the mean/std partial charges of the models' training set before evaluating the models. It can be automatically found in the log of the training, otherwise it is required to load the training set to calculate these values. When using MD data augmentation, loading the training set takes a long time.

The test set is already provided in the github.


```commandline
conda activate MD_DL_BA_env

python densenucy.py \
       training.only_test=True \
       network.pretrained_path=../../../models/densenucy/complex/densenucy_17274-complexes_5945-complexes-with-MD_59441-simulations_2925609-frames_all_with_structure.pth
```

```commandline
conda activate MD_DL_BA_env

python densenucy_systematic_rotations.py \
       training.only_test=True \
       network.pretrained_path=../../../models/densenucy_rotations/densenucy_rotations_CoG12.pth
```
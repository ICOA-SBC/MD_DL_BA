# Videonucy: ConvLSTM models used to predict binding affinity from molecular dynamics simulations of protein-ligand complexes

## Setup

* [hydra-core](https://hydra.cc/docs/intro/) is used for setting up all the options
* [mlflow](https://mlflow.org/) is used to build an experiment notebook

```commandline
conda install environment.yml
conda activate MD_DL_BA_env
```

## How to generate data listing from your own data

The goal is to build txt files containing either :

* a complex name per line

```
1y3p
4xir
187l
```

* a simulation name with its complex name per line

```
./2y7i/replicate_1.npy
./2y7i/replicate_3.npy
./4us3/replicate_8.npy
./4us3/replicate_6.npy
```

```sh
export INPUT_PATH=../datasets/4D_data
export OUTPUT_PATH=../datasets/version4D/inputs

mkdir -p $OUTPUT_PATH/{by_complex,by_sim}

# by_complex
ls $INPUT_PATH/training_set/ > $OUTPUT_PATH/by_complex/training_samples.txt
ls $INPUT_PATH/test_set/ > $OUTPUT_PATH/by_complex/test_samples.txt
ls $INPUT_PATH/validation_set/ > $OUTPUT_PATH/by_complex/validation_samples.txt

# by_sim
cd  $INPUT_PATH/training_set/
find  -type f -name '*.npy' | tee  ../../version4D/inputs/by_sim/training_samples.txt
cd  ../test_set/
find  -type f -name '*.npy' | tee  ../../version4D/inputs/by_sim/test_samples.txt
cd  ../validation_set/
find  -type f -name '*.npy' | tee  ../../version4D/inputs/by_sim/validation_samples.txt
```

If you want to use the dataset provided in the publication, you can download them on zenodo and put the training_set, validation_set and test_set at `$INPUT_PATH`

If you modified the paths, don't forget to update the paths to theses files in the `yaml` config file

## How to compute the partial charges

`main_compute_charges` will go through all the dataset samples to compute the mean and standard deviation. Don't forget to update the `yaml` config file with the values from the train dataset returned by this code!

This can take a while to compute (e.g. an hour for 60000 simulations)

```commandline
conda activate MD_DL_BA_env
python main_compute_charges.py
```

Output:
```
Computing on test dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 830/830 [00:38<00:00, 21.41it/s]
Charges : (21284400,) elements in 830 samples
Charges on dataset *** test ***: mean= -0.18674144 std= 0.44699324
Duration: 00:00:40

Computing on valid dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11668/11668 [09:02<00:00, 21.49it/s]
Charges : (300887543,) elements in 11668 samples
Charges on dataset *** valid ***: mean= -0.18212932 std= 0.45011813
Duration: 00:09:18

Computing on train dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 46632/46632 [39:19<00:00, 19.76it/s]
Charges : (1142101728,) elements in 46632 samples
Charges on dataset *** train ***: mean= -0.19065223 std= 0.45264780
        Please update your yaml file with theses values!
Duration: 00:40:27
```

## by_complex and by_sim

When the training is performed with `experiment.by_complex: true`, it means that the dataloader will go through as many items as there are complexes and for each complex, the dataset class will randomly select a simulation. It can happen that some simulations are never used during a training.

When the training is performed with `experiment.by_complex: false` (which means `by_sim` sample listings will be used), it means that the dataloader will go through each and every simulation, which will take a lot longer and requires the use of more than one GPU to be sustainable.

Performances are expected to be better with `by_sim` setup.

## Training

Training is setup to work with multi-gpu (or nodes). Nothing is automatically done for the learning_rate for the adjustement of the learning_rate (it could go up with more GPUs).

* `batch_size` is always defined per GPU.
* `LambdaLR` is a "warmup" scheduler that will start with a learning_rate of 0 up to the requested learning_rate (in the `yaml` config file). It is supposed to help get a better start for training. 
* `ReduceLROnPlateau` is another scheduler that will lower the learning_rate if the Loss doesn't improve for a while - it is usefull at the end of a training, as we may overshoot the minimum  

The `config-name` is the name of the config file, change it in case you want to use a different yaml preset file. Information about each option are displayed in the yaml file.

```commandline
conda activate MD_DL_BA_env

python videonucy_train.py \
       --config-name=videonucy \
       experiment.run=5932-complexes_58300-simulations_by-complex \
       experiment.name=convLSTM \
       network.sim_test=True \
       network.mean_test=True
```

## Testing

Metrics are computed at the end of the training in `videonucy_train.py` but it is possible to launch this computation apart with `videonucy_test.py`, it works on a single GPU.

```commandline
conda activate MD_DL_BA_env

python videonucy_test.py \
       --config-name=videonucy \
       experiment.run=5932-complexes_58300-simulations_by-complex \
       experiment.name=convLSTM \
       network.pretrained_path=../../../models/complex/convlstm_5932-complexes_58300-simulations_by-complex.pth  \
       network.sim_test=True \
       network.mean_test=True
```
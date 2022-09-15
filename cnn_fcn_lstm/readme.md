# Version4D: CNN-FCN-LSTM models for frames of protein-ligand complexes

> Note: you will probably have to update paths used in environment variables

## Setup

* hydra-core is used for setting up all the options
* mlflow is used to build an experiment notebook

> warning : they use a non negligeable amount of CPU memory and generate lots of outputs

``` sh
module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
mkdir $PYTHONUSERBASE
export PATH=$PYTHONUSERBASE/bin:$PATH

pip install --user --no-cache-dir hydra-core mlflow
```

## How to generate data listing

The goal is to build txt files containing either :

* a complex name per line

```
1y3p
4xir
187l
```

* a simulation name with its complex name per line

```
2y7i/replicate_1.npy
2y7i/replicate_3.npy
4us3/replicate_8.npy
4us3/replicate_6.npy
```

```sh
export INPUT_PATH=$dzc_ALL_CCFRWORK/deep_learning/pafnucy/data/4D_data_v3
export OUTPUT_PATH=$dzc_CCFRWORK/version4D/inputs_v3/

mkdir -p $OUTPUT_PATH/{by_complex,by_sim}

# by_complex
ls $INPUT_PATH/training_set/ > $OUTPUT_PATH/by_complex/training_samples.txt
ls $INPUT_PATH/test_set/ > $OUTPUT_PATH/by_complex/test_samples.txt
ls $INPUT_PATH/validation_set/ > $OUTPUT_PATH/by_complex/validation_samples.txt

# by_sim
cd  $INPUT_PATH/training_set/
find  -type f -name '*.npy' | tee  $OUTPUT_PATH/by_sim/training_samples.txt
cd  $INPUT_PATH/test_set/
find  -type f -name '*.npy' | tee  $OUTPUT_PATH/by_sim/test_samples.txt
cd  $INPUT_PATH/validation_set/
find  -type f -name '*.npy' | tee  $OUTPUT_PATH/by_sim/validation_samples.txt
```

Don't forget to update the paths to theses files in the `yaml` config file


## How to compute the partial charges

`main_compute_charges` will go through all the dataset samples to compute the mean and standard deviation. Don't forget to update the `yaml` config file with the values from the train dataset returned by this code!

This can take a while to compute (about 30 minutes for dataset_v3), so use a compute node for that (prepost, compil or a cpu partition).

```
export INSTALL_DIR=$dzc_CCFRWORK/version4D
cd $INSTALL_DIR

srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 --hint=nomultithread --partition=prepost  --time=02:00:00 --account dzc@v100 bash

module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

python main_compute_charges.py

```

> if there is not enough memory you can launch this code on partition prepost and/or ask more cpus_per_task

Output:

```
Version 3:
(pytorch-gpu-1.11.0+py3.9.12) bash-4.4$ python main_compute_charges.py 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 820/820 [00:42<00:00, 19.38it/s]
Charges : (21007124,) elements in 820 samples
Charges on dataset *** test ***: mean= -0.18692062 std= 0.44705638
Duration: 00:00:43
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7233/7233 [06:07<00:00, 19.66it/s]
Charges : (186706683,) elements in 7233 samples
Charges on dataset *** valid ***: mean= -0.18240544 std= 0.47032203
Duration: 00:06:20
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28907/28907 [25:29<00:00, 18.91it/s]
Charges : (697408084,) elements in 28907 samples
Charges on dataset *** train ***: mean= -0.19279409 std= 0.48856917
        Please update your yaml file with theses values!
Duration: 00:26:08

Version 4:
(pytorch-gpu-1.11.0+py3.9.12) bash-4.4$ python main_compute_charges.py
Computing on test dataset -------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 830/830 [00:35<00:00, 23.35it/s]
Charges : (21019844,) elements in 830 samples
Charges on dataset *** test ***: mean= -0.18746309 std= 0.44717112
Duration: 00:00:36

Computing on valid dataset -------------
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10395/10395 [07:24<00:00, 23.38it/s]
Charges : (259859568,) elements in 10395 samples
Charges on dataset *** valid ***: mean= -0.18522244 std= 0.46666227
Duration: 00:07:38

Computing on train dataset -------------
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41544/41544 [29:12<00:00, 23.71it/s]
Charges : (966421624,) elements in 41544 samples
Charges on dataset *** train ***: mean= -0.19436523 std= 0.65582558
	Please update your yaml file with theses values!
Duration: 00:30:01
```

## Model architecture

### Small version

Basically, the model doesn't duplicate the CNN-LSTM part: all the frames will go through a unique CNN. Then each output (as many as the frame number) will be concatenated and will feed the LSTM layers.

Basically, the CNN-LSTM part should be able to predict the affinity and the LSTM part add an interpretation level to this sequence of affinity to output a single prediction.

The CNN-LSTM part is inspired by the models used on 3D data:
* `cnn.py` contains regular Conv3d layers
* `densecnn.py` contains DenseBlock (as found in **densenet**)

### Large version (not operational)

> empty

## by_complex and by_sim

When the training is performed with `experiment.by_complex: true`, it means that the dataloader will go through as many items as there are complexes and for each complex, the dataset class will randomly select a simulation. It can happen that some simulations are never used during a training.

When the training is performed with `experiment.by_complex: false` (which means `by_sim` sample listings will be used), it means that the dataloader will go through each and every simulation, which will take a lot longer and requires the use of more than one GPU to be sustainable.

Performances are expected to be better with `by_sim` setup.

## Testing the dataloader and the model (no training)

Please use `main_test_dataset.py`.

```sh
export INSTALL_DIR=$dzc_CCFRWORK/version4D
cd $INSTALL_DIR

srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread      --qos=qos_gpu-t3 --time=01:00:00 --account sos@v100 bash

module load pytorch-gpu/py3/1.11.0
export PYTHONUSERBASE=$WORK/.local_convlstm
export PATH=$PYTHONUSERBASE/bin:$PATH

python main_test_dataset.py

```

> This code is only useful to check changes in the model or the dataset class

## Training

Training is setup to work with multi-gpu (or nodes). Nothing is automatically done for the learning_rate for the adjustement of the learning_rate (it could go up with more GPUs).

* `batch_size` is always defined per GPU.
* `LambdaLR` is a "warmup" scheduler that will start with a learning_rate of 0 up to the requested learning_rate (in the `yaml` config file). It is supposed to help get a better start for training. 
* `ReduceLROnPlateau` is another scheduler that will lower the learning_rate if the Loss doesn't improve for a while - it is usefull at the end of a training, as we may overshoot the minimum  




## Testing

Metrics are computed at the end of the training in `main_train.py` but it is possible to launch this computation aprt with `main_test.py`, it works on a single GPU.


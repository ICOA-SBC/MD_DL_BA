import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from codes.tools import convert_time

feature_names = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode',
                 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
columns = {name: 3 + i for i, name in enumerate(feature_names)}  # skip 'x', 'y', 'z'


def get_samples_list(io, pathname, filename, by_complex=False):
    if by_complex:
        filename = os.path.join(io.by_complex_dir, filename)
    else:
        filename = os.path.join(io.by_sim_dir, filename)

    with open(filename, 'r') as f:
        samples_list = [line.rstrip() for line in f.readlines()]

    samples_list = [os.path.join(pathname, s) for s in samples_list]
    return samples_list


def get_mean_charges_per_simulation(sim):
    charges, dim = 0.0, 0
    for frame in sim:
        col = frame[:, columns['partialcharge']]
        charges += sum(col)
        dim += len(col)
    return charges / dim


def get_charges_per_simulation(sim):
    charges = []
    for frame in sim:
        col = frame[:, columns['partialcharge']]
        charges.extend(col)
    return charges


@hydra.main(config_path="./configs", config_name="default_datasetv3")
def main(cfg: DictConfig) -> None:
    """
    Compute charge mean and std over the whole train dataset (63Go)
    Read each and every sim and each frame per sim
    """
    by_complex = False  # overrides config file

    for name in ['test', 'valid', 'train']:
        start_time = time.time()

        print(f"Computing on {name} dataset -------------")
        if name == 'test':
            samples = get_samples_list(cfg.io, cfg.io.test_dir, cfg.io.test_samples, by_complex)
        elif name == 'valid':
            samples = get_samples_list(cfg.io, cfg.io.val_dir, cfg.io.val_samples, by_complex)
        else:
            samples = get_samples_list(cfg.io, cfg.io.train_dir, cfg.io.train_samples, by_complex)

        charges = []
        for i, s in enumerate(tqdm(samples)):
            sim = np.load(s, allow_pickle=True)
            charges.extend(get_charges_per_simulation(sim))

        charges = np.array(charges)
        print(f"Charges : {charges.shape} elements in {len(samples)} samples")
        charges_mean, charges_std = charges.mean(), charges.std()
        print(f"Charges on dataset *** {name} ***: mean= {charges_mean:.8f} std= {charges_std:.8f}")
        if name == 'train':
            print("\tPlease update your yaml file with theses values!")

        print(f"Duration: {convert_time(time.time() - start_time)}\n")


if __name__ == "__main__":
    main()

'''
dataset v2

(pytorch-gpu-1.11.0+py3.9.12) bash-4.4$ python main_compute_charges.py 
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [00:33<00:00, 24.07it/s]
Charges : (20561451,) elements in 800 samples
Charges on dataset Train: mean= -0.18533515 std= 0.44652179
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6128/6128 [04:12<00:00, 24.26it/s]
Charges : (155825022,) elements in 6128 samples
Charges on dataset Train: mean= -0.18310309 std= 0.45129641
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24484/24484 [16:51<00:00, 24.19it/s]
Charges : (590542187,) elements in 24484 samples
Charges on dataset Train: mean= -0.19319045 std= 0.45451332
'''

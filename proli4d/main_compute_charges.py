import os

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

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


@hydra.main(config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Compute charge mean and std over the whole train dataset (63Go)
    Read each and every sim and each frame per sim
    """
    by_complex = False  # overrides config file
    # samples = get_samples_list(cfg.io, cfg.io.train_dir, cfg.io.train_samples, by_complex)
    samples = get_samples_list(cfg.io, cfg.io.test_dir, cfg.io.test_samples, by_complex)
    if cfg.debug:
        print(f"Reading {len(samples)} samples")

    charges = []
    for i, s in enumerate(tqdm(samples)):
        sim = np.load(s, allow_pickle=True)
        charges.extend(get_charges_per_simulation(sim))

    charges = np.array(charges)
    print(f"Charges : {charges.shape} elements in {len(samples)} samples")
    charges_mean, charges_std = charges.mean(), charges.std()
    print(f"Charges on dataset Train: mean= {charges_mean:.8f} std= {charges_std:.8f}")


if __name__ == "__main__":
    main()

"""
Results :
(pytorch-gpu-1.11.0+py3.9.12) bash-4.4$ python main_compute_charges.py 
Reading 15836 samples
100%|██████████████████████████████████████| 15836/15836 [10:27<00:00, 25.25it/s]
Charges : (377729879,) elements in 15836 samples
Charges on dataset Train: mean= -0.19513957 std= 0.45485657

Reading 790 samples
100%|██████████████████████████████████████| 790/790 [00:32<00:00, 24.38it/s]
Charges : (20307671,) elements in 790 samples
Charges on dataset Test: mean= -0.18548983 std= 0.44664063



Data set original (3D)
partialcharge:
  m: -0.1401471346616745
  std: 0.4216829240322113 

"""

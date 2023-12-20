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


@hydra.main(config_path="./configs", config_name="videonucy")
def main(cfg: DictConfig) -> None:
    """
    Compute charge mean and std over the whole train dataset (63Go)
    Read each and every sim and each frame per sim
    """
    by_complex = False  # overrides config file
    
    for name in ['test', 'valid', 'train']:
        if name == 'test':
            samples = get_samples_list(cfg.io, cfg.io.test_dir, cfg.io.test_samples, by_complex)
        elif name == 'valid':
            samples = get_samples_list(cfg.io, cfg.io.val_dir, cfg.io.val_samples, by_complex)
        else:
            samples = get_samples_list(cfg.io, cfg.io.train_dir, cfg.io.train_samples, by_complex)
    
   
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
V7:
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



V7 only ligand:
Computing on test dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 830/830 [00:11<00:00, 69.63it/s]
Charges : (952500,) elements in 830 samples
Charges on dataset *** test ***: mean= -0.10649685 std= 0.36160164
Duration: 00:00:12

Computing on valid dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11665/11665 [03:10<00:00, 61.10it/s]
Charges : (15329100,) elements in 11665 samples
Charges on dataset *** valid ***: mean= -0.12943018 std= 0.44109652
Duration: 00:03:11

Computing on train dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 46634/46634 [16:14<00:00, 47.83it/s]
Charges : (75665750,) elements in 46634 samples
Charges on dataset *** train ***: mean= -0.12487510 std= 0.43172313
        Please update your yaml file with theses values!
Duration: 00:16:18



V7 only protein:
Computing on test dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 830/830 [00:36<00:00, 22.48it/s]
Charges : (20331900,) elements in 830 samples
Charges on dataset *** test ***: mean= -0.19050071 std= 0.45024644
Duration: 00:00:38

Computing on valid dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11665/11665 [08:35<00:00, 22.65it/s]
Charges : (284080181,) elements in 11665 samples
Charges on dataset *** valid ***: mean= -0.18508366 std= 0.45048237
Duration: 00:08:49

Computing on train dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 46657/46657 [35:18<00:00, 22.02it/s]
Charges : (1068300111,) elements in 46657 samples
Charges on dataset *** train ***: mean= -0.19528796 std= 0.45371858
        Please update your yaml file with theses values!
Duration: 00:36:13



V7 only ligand old:
Computing on test dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 830/830 [00:13<00:00, 60.51it/s]
Charges : (952500,) elements in 830 samples
Charges on dataset *** test ***: mean= -0.10649685 std= 0.36160164
Duration: 00:00:13

Computing on valid dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11857/11857 [04:09<00:00, 47.59it/s]
Charges : (15325500,) elements in 11857 samples
Charges on dataset *** valid ***: mean= -0.12872209 std= 0.43997518
Duration: 00:04:09

Computing on train dataset -------------
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 47385/47385 [18:58<00:00, 41.62it/s]
Charges : (77086250,) elements in 47385 samples
Charges on dataset *** train ***: mean= -0.12479054 std= 0.43175010
        Please update your yaml file with theses values!
Duration: 00:19:02
"""

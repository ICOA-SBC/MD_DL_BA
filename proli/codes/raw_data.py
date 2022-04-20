from os.path import join

import h5py
from numpy import array, float32, concatenate

feature_names = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal', \
                 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode', \
                 'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
columns = {name: i for i, name in enumerate(feature_names)}


class RawDataset():

    def __init__(self, input_dir, name, max_dist=None, verbose=False):
        self.name = name
        self.dataset_fullpath = join(input_dir, f"{name}_set.hdf")
        self.max_dist = max_dist
        self.verbose = verbose
        self.ids = None
        self.affinity = None
        self.coords = []
        self.features = []
        self.max_coord = []
        self.columns = None
        self.charges_mean = None
        self.charges_std = None

    def __len__(self):
        return len(self.affinity)

    def __str__(self):
        description = f"{self.name} dataset with {self.__len__()} samples\n"
        description += f"\tPartial charge normalization: m= {self.charges_mean} \
                    std= {self.charges_std}\n"

        return description

    def load_data(self):
        ids, affinity = [], []

        with h5py.File(self.dataset_fullpath, "r") as f:
            for pdb_id in f:
                data = f[pdb_id]
                self.coords.append(data[:, :3])
                self.features.append(data[:, 3:])
                ids.append(pdb_id)
                affinity.append(data.attrs['affinity'])

        self.ids = array(ids)
        self.affinity = array(affinity, dtype=float32)

        # TODO understand what should happen when max_dist is undefined !!!
        # if not self.max_dist:
        #    self.max_coord.append(np.max(self.coords))  # incorrect
        # 
        if self.name == "training":
            self.compute_normalization_params()

    def compute_normalization_params(self):
        charges = []
        for data in self.features:
            charges.append(data[..., columns['partialcharge']])
        charges = concatenate([c.flatten() for c in charges])
        self.charges_mean = charges.mean()
        self.charges_std = charges.std()
        return self.charges_mean, self.charges_std

    def set_normalization_params(self, mean_val, std_val):
        self.charges_mean = mean_val
        self.charges_std = std_val

    def get_normalization_params(self):
        return self.charges_mean, self.charges_std

    # New: normalization is done on raw data
    def charge_normalization(self):
        for data in self.features:
            data[..., columns['partialcharge']] = \
                (data[..., columns['partialcharge']] - self.charges_mean) / self.charges_std

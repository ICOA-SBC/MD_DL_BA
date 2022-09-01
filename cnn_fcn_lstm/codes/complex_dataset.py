import os
from random import sample

import numpy as np
from numpy.random import randint
from torch import from_numpy
from torch.utils.data import Dataset

from codes.frame import Frame


class Complexes_4DDataset(Dataset):
    """
    Format of a sample :
        dataset -> complex -> simulation -> 50 frames
    """

    def __init__(self, inputs, setup, by_complex=True, rotations=None, mode='train', debug=False):
        self.max_dist = setup.max_dist
        self.grid_spacing = setup.grid_spacing
        self.by_complex = by_complex  # if true, then a sim is randomly picked when a complex is selected (__getitem__)
        self.mode = mode.lower()  # either train, val or test
        self.rotations = rotations
        self.number_of_rotations = len(self.rotations) if rotations else 0

        self.samples_list = self.read_samples_list(inputs)
        self.affinities = self.read_affinities(inputs.affinities)

        self.charges_mean = setup.partial_charges.mean
        self.charges_std = setup.partial_charges.std
        self.frames = setup.frames  # how many frames are kept
        self.keep_frames_in_order = setup.keep_frames_in_order
        self.debug = debug

    def read_samples_list(self, inputs):
        if self.mode.startswith('train'):
            filename, path = inputs.train_samples, inputs.train_dir
        elif self.mode.startswith('val'):
            filename, path = inputs.val_samples, inputs.val_dir
        else:
            filename, path = inputs.test_samples, inputs.test_dir

        if self.by_complex:
            filename = os.path.join(inputs.by_complex_dir, filename)
        else:
            filename = os.path.join(inputs.by_sim_dir, filename)

        with open(filename, 'r') as f:
            samples_list = [line.rstrip() for line in f.readlines()]

        samples_list = [os.path.join(path, s) for s in samples_list]

        return samples_list

    def read_affinities(self, affinities_filenames):
        aff = []
        for f in affinities_filenames:
            with open(f, 'r') as f:
                data = [line.rstrip() for line in f.readlines() if not line.startswith("#")]
            k = [self.get_pdb_aff(sample) for sample in data]
            aff.extend(k)

        affinities = dict((a[0], a[1]) for a in aff)

        return affinities

    def set_normalization_params(self, mean_val, std_val):
        self.charges_mean = mean_val
        self.charges_std = std_val

    def get_normalization_params(self):  # TODO : useful ?
        return self.charges_mean, self.charges_std

    @staticmethod
    def get_pdb_aff(pdb_set_sample):
        # retrieve pbd code and -logKd/Ki (1st and 4th column)
        pdb, _, _, aff, *_ = pdb_set_sample.split()
        return pdb, float(aff)

    @staticmethod
    def get_pdb_name(selected_sample):
        p = selected_sample
        if selected_sample.endswith("npy"):
            p = os.path.dirname(selected_sample)  # remove filename

        return os.path.basename(p)

    def load_sim(self, idx):
        # A simulation contains 50 frames
        sample_filename = self.samples_list[idx]
        pdb_name = self.get_pdb_name(sample_filename)

        if self.by_complex:
            # randomly select one sim
            list_of_sims = [f for f in os.listdir(sample_filename) if f.endswith('.npy')]
            target = randint(0, len(list_of_sims))  # a <= N <= b
            sample_filename = os.path.join(sample_filename, list_of_sims[target])

        frames = np.load(sample_filename, allow_pickle=True)

        if self.debug:
            print(f"Loading sample: {sample_filename}")
            print(f"\tdata format: {frames.shape}")

        affinity = self.affinities[pdb_name]
        return frames, affinity

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        # get sample in raw format (pick a random sim if relevant)
        frames, affinity = self.load_sim(idx)
        selected_rotation = None
        if self.rotations:
            selected_rotation = self.rotations[randint(self.number_of_rotations)]
            # print(f"rotation {selected_rotation.shape}")
        inputs = []
        frame = Frame(selected_rotation, self.charges_mean, self.charges_std, self.grid_spacing, self.max_dist)
        selected_frames = sample(list(range(len(frames))), self.frames)
        if self.keep_frames_in_order:
            selected_frames = sorted(selected_frames)
        for f in selected_frames:
            volume = frame.get_frame(frames[f])
            inputs.append(volume)
        inputs = np.stack(inputs, axis=0)
        inputs = from_numpy(inputs)
        return inputs, affinity

from math import ceil

import numpy as np


class Frame:
    feature_names = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'halogen', 'metal',
                     'hyb', 'heavyvalence', 'heterovalence', 'partialcharge', 'molcode',
                     'hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
    cols = {name: i for i, name in enumerate(feature_names)}

    def __init__(self, selected_rotation=None, charges_mean=.0, charges_std=None, grid_spacing=1.0, max_dist=10.0):
        self.selected_rotation = selected_rotation
        self.charges_mean = charges_mean
        self.charges_std = charges_std
        self.grid_spacing = grid_spacing
        self.max_dist = max_dist

    @staticmethod
    def convert_to_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
        coords = np.asarray(coords, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        num_features = features.shape[1]
        max_dist = float(max_dist)
        grid_resolution = float(grid_resolution)

        box_size = ceil(2 * max_dist / grid_resolution + 1)

        # move all atoms to the nearest grid point
        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)

        # remove atoms outside the box
        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        volume = np.zeros((box_size, box_size, box_size, num_features), dtype=np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            volume[x, y, z] += f

        return volume

    def get_frame(self, frame):
        coords, features = frame[:, :3], frame[:, 3:]

        if self.selected_rotation is not None:
            coords = np.dot(coords, self.selected_rotation)
        if self.charges_std:
            features[:, Frame.cols['partialcharge']] = \
                (features[:, Frame.cols['partialcharge']] - self.charges_mean) / self.charges_std

        volume = self.convert_to_grid(coords, features, self.grid_spacing, self.max_dist)
        volume = np.moveaxis(volume, -1, 0)  # (25,25,25,19)--> (19,25,25,25)
        return volume

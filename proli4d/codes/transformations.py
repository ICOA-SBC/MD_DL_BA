from itertools import combinations
from math import sqrt, cos, sin, pi

import numpy as np


def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis = np.asarray(axis, dtype=np.float32)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def build_rotations():
    # Create matrices for all possible 90* rotations of a box
    rotations_matrices = [rotation_matrix([1, 1, 1], 0)]

    # about X, Y and Z - 9 rotations
    for a1 in range(3):
        for t in range(1, 4):
            axis = np.zeros(3)
            axis[a1] = 1
            theta = t * pi / 2.0
            rotations_matrices.append(rotation_matrix(axis, theta))

    # about each face diagonal - 6 rotations
    for (a1, a2) in combinations(range(3), 2):
        axis = np.zeros(3)
        axis[[a1, a2]] = 1.0
        theta = pi
        rotations_matrices.append(rotation_matrix(axis, theta))
        axis[a2] = -1.0
        rotations_matrices.append(rotation_matrix(axis, theta))

    # about each space diagonal - 8 rotations
    for t in [1, 2]:
        theta = t * 2 * pi / 3
        axis = np.ones(3)
        rotations_matrices.append(rotation_matrix(axis, theta))
        for a1 in range(3):
            axis = np.ones(3)
            axis[a1] = -1
            rotations_matrices.append(rotation_matrix(axis, theta))

    return rotations_matrices

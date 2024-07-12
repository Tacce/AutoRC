from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from sequence import Sequence
import numpy as np


def calculate_trajectory_length(points):
    return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))


angle_threshold = np.pi / 16
max_slope = 6


def classify_trajectory(points):
    # Considero solo la traiettoria sul piano xz
    points = points[:, [0, 2]]

    diffs = np.diff(points, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_diffs = np.diff(angles)

    # Normalizza gli angoli in [-pi, pi]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

    total_angle = np.sum(angle_diffs)

    slope = (points[-1, 1] - points[0, 1]) / (points[-1, 0] - points[0, 0]) if points[-1, 0] != points[0, 0] else np.inf

    if abs(slope) > max_slope:
        return 'straight'

    if np.abs(total_angle) < angle_threshold:
        if slope > 2:
            return 'straight'
        elif slope > 0:
            return 'right'
        else:
            return 'left'
    elif total_angle > 0:
        return 'left'
    else:
        return 'right'


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

for delta_f in range(5, 21, 5):

    stats = defaultdict(list)
    stats['n_examples'] = 0
    stats['straight'] = 0
    stats['left'] = 0
    stats['right'] = 0
    # stats['maxdi']

    for s in sequences:
        for t in range(0, s.lenght - delta_f, 3):
            sample = s.get_sample(t, delta_f, 0)
            trajectory = sample[1]

            stats['trajectory_list'].append(trajectory[:, [0, 2]])
            stats[classify_trajectory(trajectory)] += 1
            stats['n_examples'] += 1
            stats['trajectory_length'].append(calculate_trajectory_length(trajectory))
            # max_distance = np.max(np.linalg.norm(np.asarray(point_cloud.points), axis=1))

    for trajectory in stats['trajectory_list']:
        plt.plot(trajectory[:, 1], -trajectory[:, 0])
    plt.title(f'{delta_f}-frame long trajectories')
    plt.axis('equal')
    plt.show()

    print(f'{delta_f}-FRAME LONG TRAJECTORY')
    print(f"Numero esempi: {stats['n_examples']}")
    print(f"Numero traitettorie rettilinee: {stats['straight']}")
    print(f"Numero curve a sinistra: {stats['left']}")
    print(f"Numero curve a destra: {stats['right']}")
    print(f"Media lunghezza traiettorie: {np.mean(stats['trajectory_length'])}")
    print(f"Varianza lunghezza traiettorie: {np.var(stats['trajectory_length'])}")
    print("______________________________________________________________________")

from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from sequence import Sequence
import numpy as np


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

for delta_f in range(5, 21, 5):

    stats = defaultdict(list)
    stats['n_examples'] = 0
    stats['straight'] = 0
    stats['left'] = 0
    stats['right'] = 0

    for s in sequences:
        for t in range(0, s.lenght - delta_f, 3):
            sample = s.get_sample(t, delta_f, 0)

            stats['trajectory_list'].append(sample.future_trajectory[:, [0, 2]])
            stats[sample.classify_trajectory()] += 1
            stats['n_examples'] += 1
            stats['trajectory_length'].append(sample.calculate_trajectory_length())
            stats['covered_area'].append(sample.calculate_covered_area_percentage())

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
    print(f"Media area coperta dalla point-cloud: {np.mean(stats['covered_area'])} %")
    print(f"Varianza area coperta dalla point-cloud: {np.var(stats['covered_area'])} %")
    print("______________________________________________________________________")

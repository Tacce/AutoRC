from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from sequence import Sequence
import numpy as np

sample_step = 2
num_bins = 7


def remove_outliers(samples):
    # Calcola la lunghezza di ogni traiettoria
    lengths = np.array([sample.get_trajectory_length() for sample in samples])

    # Calcola i quartili e l'IQR
    Q1 = np.percentile(lengths, 25)
    Q3 = np.percentile(lengths, 75)
    IQR = Q3 - Q1

    # Calcola la soglia superiore
    upper_threshold = Q3 + 1.5 * IQR
    lower_threshold = Q1 - 1.5 * IQR

    # Filtra i sample che hanno una traiettoria che supera la soglia superiore
    filtered_samples = [sample for sample, length in zip(samples, lengths)
                        if upper_threshold >= length >= lower_threshold]

    return filtered_samples


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

for delta_f in range(2, 9, 2):

    samples = []

    stats = defaultdict(list)
    stats['n_examples'] = 0
    stats['straight'] = 0
    stats['left'] = 0
    stats['right'] = 0

    for s in sequences:
        for t in range(0, int(s.lenght - delta_f), sample_step):
            sample = s.get_sample(t, delta_f, 0)
            samples.append(sample)

    samples = remove_outliers(samples)

    for sample in samples:
        stats['trajectory_list'].append(sample.future_trajectory)
        stats[sample.classify_trajectory()] += 1
        stats['n_examples'] += 1
        stats['trajectory_length'].append(sample.get_trajectory_length())
        stats['covered_area'].append(sample.calculate_covered_area_percentage())

    for trajectory in stats['trajectory_list']:
        plt.plot(trajectory.points[:, 2], -trajectory.points[:, 0])
    plt.title(f'{delta_f} seconds long trajectories')
    plt.axis('equal')
    plt.show()

    max_distances = [s.calculate_max_distance_from_origin() for s in samples]
    plt.hist(max_distances, bins=num_bins, edgecolor='black')
    plt.title('Histogram of the Maximum Distances from the Point Cloud to the Origin')
    plt.xlabel('Maximum Distance')
    plt.ylabel('Frequency')
    plt.show()

    print(f'{delta_f} SECONDS LONG TRAJECTORIES')
    print(f"Numero esempi: {stats['n_examples']}")
    print(f"Numero traitettorie rettilinee: {stats['straight']}")
    print(f"Numero curve a sinistra: {stats['left']}")
    print(f"Numero curve a destra: {stats['right']}")
    print(f"Media lunghezza traiettorie: {np.mean(stats['trajectory_length'])} m")
    print(f"Varianza lunghezza traiettorie: {np.var(stats['trajectory_length'])}")
    print(f"Media area coperta dalla point-cloud: {np.mean(stats['covered_area'])} %")
    print(f"Varianza area coperta dalla point-cloud: {np.var(stats['covered_area'])}")
    print("______________________________________________________________________\n")

from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from sequence import Sequence
import numpy as np

sample_step = 2
num_bins = 10
framerate = 10
max_velocity = 2.8


def plot_histogram(data, num_bins, title, xlabel):
    plt.hist(data, bins=num_bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()


def remove_outliers_length(samples):
    # Calcola la lunghezza di ogni traiettoria
    lengths = np.array([sample.get_trajectory_length() for sample in samples])

    # Calcola i quartili e l'IQR
    Q1 = np.percentile(lengths, 25)
    Q3 = np.percentile(lengths, 75)
    IQR = Q3 - Q1

    # Calcola la soglia superiore
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Filtra i sample che hanno una traiettoria che supera la soglia superiore
    filtered_samples = [sample for sample, length in zip(samples, lengths)
                        if upper_bound >= length >= lower_bound]

    return filtered_samples


def remove_outliers_consecutive_distances(samples):
    filtered_samples = [sample for sample in samples if not sample.is_outlier()]
    print(f"{len(samples) - len(filtered_samples)} outliers removed")
    return filtered_samples


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

for delta_f in range(2, 9, 2):
    delta_p = 2

    samples = []

    stats = defaultdict(list)
    stats['n_examples'] = 0
    stats['straight'] = 0
    stats['left'] = 0
    stats['right'] = 0

    for s in sequences:
        stats['start_angles'].append(s.calcuate_start_angle())
        stats['time_length'].append(s.lenght)
        stats['num_frames'].append(s.num_frames)
        for t in range(s.num_frames):
            sample = s.get_sample(t, delta_f, delta_p, framerate, max_velocity)
            if sample is not None:
                samples.append(sample)
                stats['trajectory_list'].append(sample.future_trajectory)
                stats[sample.classify_trajectory()] += 1
                stats['n_examples'] += 1
                stats['trajectory_length'].append(sample.get_trajectory_length())
                stats['covered_area'].append(sample.calculate_covered_area_percentage())
                stats['distances'].extend(sample.calculate_trajectory_points_distances())
                stats['framerates'].append(sample.original_framerate)

                curve = sample.calculate_curve_angle()
                if curve is not None:
                    stats['curve_angles'].append(curve)

    for trajectory in stats['trajectory_list']:
        plt.plot(trajectory.points[:, 2], -trajectory.points[:, 0])
    plt.title(f'{delta_f} seconds long trajectories')
    plt.axis('equal')
    plt.show()

    max_distances = [s.calculate_max_distance_from_origin() for s in samples]
    plot_histogram(max_distances, num_bins, 'Histogram of the Maximum Distances from the Point Cloud to the Origin',
                   'Maximum Distance')

    plot_histogram(stats['trajectory_length'], num_bins, 'Histogram of the Trajectory Lengths', 'Trajectory Length')

    plot_histogram(stats['covered_area'], num_bins, 'Histogram of the Percentage of Area Covered by the Point Cloud',
                   'Covered Area Percentage')

    plot_histogram(stats['framerates'], num_bins, 'Histogram of the Original Framerates', 'Framerate')

    plot_histogram(stats['distances'], 100, 'Distance Distribution', 'Distance')

    plot_histogram(stats['start_angles'], 20, 'Start Angles', 'Start Angles')

    plot_histogram(stats['curve_angles'], 50, 'Curve Angles Distribution', 'Curve Angles')

    print(f'{delta_f} SECONDS LONG TRAJECTORIES')
    print(f"Numero esempi: {stats['n_examples']}")
    print(f"Lunghezza media sequenze: {np.mean(stats['time_length'])} s")
    print(f"Numero medio frame: {np.mean(stats['num_frames'])}")
    print(f"Numero traitettorie rettilinee: {stats['straight']}")
    print(f"Numero curve a sinistra: {stats['left']}")
    print(f"Numero curve a destra: {stats['right']}")
    print(f"Media lunghezza traiettorie: {np.mean(stats['trajectory_length'])} m")
    print(f"Varianza lunghezza traiettorie: {np.var(stats['trajectory_length'])}")
    print(f"Media framerate: {np.mean(stats['framerates'])}")
    print(f"Varianza framerate: {np.var(stats['framerates'])}")
    print(f"Media area coperta dalla point-cloud: {np.mean(stats['covered_area'])} %")
    print(f"Varianza area coperta dalla point-cloud: {np.var(stats['covered_area'])}")
    print(f"Media distanza punti: {np.mean(stats['distances'])}")
    print(f"Varianza distanza punti: {np.var(stats['distances'])}")
    print(f"Media angolo d'attacco: {np.mean(stats['start_angles'])}°")
    print(f"Varianza angolo d'attacco: {np.var(stats['start_angles'])}")
    print(f"Media angolo di curva: {np.mean(stats['curve_angles'])}°")
    print(f"Varianza angolo di curva: {np.var(stats['curve_angles'])}")
    print("______________________________________________________________________\n")

from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from sequence import Sequence
import numpy as np


# Framerate imposto (il numero di punti della traiettoria risulta essere framerate * delta_f)
framerate = 10
# Velocità massima consentita al veicolo per considerare il campione valido
max_velocity = 2.8
# Parametro che indica ogni quanti campioni plottare la traiettoria (per evitare confusione)
plot_step = 10

delta_p = 2
# Numero standard per colonne degli istogrammi
num_bins = 15


def plot_histogram(data, num_bins, title, xlabel, save_path):
    plt.hist(data, bins=num_bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    upper_limit = np.percentile(data, 99)
    lower_limit = np.percentile(data, 1)
    plt.xlim(lower_limit, upper_limit)

    plt.savefig(save_path)
    plt.show()
    plt.close()


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

for delta_f in range(2, 7, 2):

    samples = []

    stats = defaultdict(list)
    stats['n_examples'] = 0
    stats['straight'] = 0
    stats['left'] = 0
    stats['right'] = 0

    for s in sequences:
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

    output_dir = Path(f"dataset_delta_f_{delta_f}")
    output_dir.mkdir(exist_ok=True)

    for trajectory in stats['trajectory_list'][::plot_step]:
        plt.plot(trajectory.points[:, 2], -trajectory.points[:, 0], 'o')
    plt.title(f'{delta_f} seconds long trajectories')
    plt.axis('equal')
    plt.savefig(output_dir / f'{delta_f}_seconds_long_trajectories.png')
    plt.show()
    plt.close()

    max_distances = [s.calculate_max_distance_from_origin() for s in samples]
    plot_histogram(max_distances, num_bins, 'Maximum Distances from the Point Cloud to the Origin',
                   'Maximum Distance', output_dir / 'histogram_max_distances.png')

    plot_histogram(stats['trajectory_length'], num_bins, 'Trajectory Lengths', 'Trajectory Length',
                   output_dir / 'histogram_trajectory_lengths.png')

    plot_histogram(stats['covered_area'], num_bins, 'Percentage of Area Covered by the Point Cloud',
                   'Covered Area Percentage', output_dir / 'histogram_covered_area.png')

    plot_histogram(stats['framerates'], num_bins, 'Original Framerates', 'Framerate',
                   output_dir / 'histogram_framerates.png')

    plot_histogram(stats['distances'], 100, 'Distance Distribution', 'Distance',
                   output_dir / 'histogram_distances.png')

    plot_histogram(stats['curve_angles'], 50, 'Curve Angles Distribution', 'Curve Angles',
                   output_dir / 'histogram_curve_angles.png')

    with open(output_dir / 'stats.txt', 'w') as f:
        print(f'{delta_f} SECONDS LONG TRAJECTORIES')
        f.write(f'{delta_f} SECONDS LONG TRAJECTORIES\n')
        print(f"Numero esempi: {stats['n_examples']}")
        f.write(f"Numero esempi: {stats['n_examples']}\n")
        print(f"Numero traitettorie rettilinee: {stats['straight']}")
        f.write(f"Numero traitettorie rettilinee: {stats['straight']}\n")
        print(f"Numero curve a sinistra: {stats['left']}")
        f.write(f"Numero curve a sinistra: {stats['left']}\n")
        print(f"Numero curve a destra: {stats['right']}")
        f.write(f"Numero curve a destra: {stats['right']}\n")
        print(f"Media lunghezza traiettorie: {np.mean(stats['trajectory_length'])} m")
        f.write(f"Media lunghezza traiettorie: {np.mean(stats['trajectory_length'])} m\n")
        print(f"Varianza lunghezza traiettorie: {np.var(stats['trajectory_length'])}")
        f.write(f"Varianza lunghezza traiettorie: {np.var(stats['trajectory_length'])}\n")
        print(f"Media framerate: {np.mean(stats['framerates'])}")
        f.write(f"Media framerate: {np.mean(stats['framerates'])}\n")
        print(f"Varianza framerate: {np.var(stats['framerates'])}")
        f.write(f"Varianza framerate: {np.var(stats['framerates'])}\n")
        print(f"Media area coperta dalla point-cloud: {np.mean(stats['covered_area'])} %")
        f.write(f"Media area coperta dalla point-cloud: {np.mean(stats['covered_area'])} %\n")
        print(f"Varianza area coperta dalla point-cloud: {np.var(stats['covered_area'])}")
        f.write(f"Varianza area coperta dalla point-cloud: {np.var(stats['covered_area'])}\n")
        print(f"Media distanza punti: {np.mean(stats['distances'])}")
        f.write(f"Media distanza punti: {np.mean(stats['distances'])}\n")
        print(f"Varianza distanza punti: {np.var(stats['distances'])}")
        f.write(f"Varianza distanza punti: {np.var(stats['distances'])}\n")
        print(f"Media angolo di curva: {np.mean(stats['curve_angles'])}°")
        f.write(f"Media angolo di curva: {np.mean(stats['curve_angles'])}°\n")
        print(f"Varianza angolo di curva: {np.var(stats['curve_angles'])}")
        f.write(f"Varianza angolo di curva: {np.var(stats['curve_angles'])}\n")
        print("______________________________________________________________________\n")

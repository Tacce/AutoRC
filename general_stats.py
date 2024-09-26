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
num_bins = 10


def plot_histogram(data, num_bins, title, xlabel, save_path):
    plt.hist(data, bins=num_bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.show()
    plt.close()


sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))


stats = defaultdict(list)
stats['better_quality_sequences'] = 0
stats['low_quality_sequences'] = 0
for s in sequences:
    stats['start_angles'].append(s.calcuate_start_angle())
    stats['time_length'].append(s.lenght)
    stats['num_frames'].append(s.num_frames)
    stats['framerate'].append(s.num_frames/s.lenght)

    if s.low_quality:
        stats['low_quality_lenght'].append(s.lenght)
        stats['low_quality_sequences'] += 1
    else:
        stats['better_quality_lenght'].append(s.lenght)
        stats['better_quality_sequences'] += 1

output_dir = Path("general_stats")
output_dir.mkdir(exist_ok=True)

plot_histogram(stats['start_angles'], 20, 'Start Angles', 'Start Angles',
               output_dir/'histogram_start_angles.png')
plot_histogram(stats['framerate'], 15, 'Sequence Framerate', 'Framerate',
               output_dir/'histogram_framerate.png')

with open('stats.txt', 'w') as f:
    print(f"Numero sequenze a bassa qualità: {stats['low_quality_sequences']}")
    f.write(f"Numero sequenze a bassa qualità: {stats['low_quality_sequences']}\n")
    print(f"Numero sequenze a qualità superiore: {stats['better_quality_sequences']}")
    f.write(f"Numero sequenze a qualità superiore: {stats['better_quality_sequences']}\n")
    print(f"Lunghezza totale sequenze: {np.sum(stats['time_length'])} s")
    f.write(f"Lunghezza totale sequenze: {np.sum(stats['time_length'])} s\n")
    print(f"Lunghezza media sequenze: {np.mean(stats['time_length'])} s")
    f.write(f"Lunghezza media sequenze: {np.mean(stats['time_length'])} s\n")
    print(f"Lunghezza totale clip a bassa qualità : {np.sum(stats['low_quality_lenght'])} s")
    f.write(f"Lunghezza totale clip a bassa qualità : {np.sum(stats['low_quality_lenght'])} s\n")
    print(f"Lunghezza totale clip a qualità superiore: {np.sum(stats['better_quality_lenght'])} s")
    f.write(f"Lunghezza totale clip a qualità superiore: {np.sum(stats['better_quality_lenght'])} s\n")
    print(f"Numero medio frame: {np.mean(stats['num_frames'])}")
    f.write(f"Numero medio frame: {np.mean(stats['num_frames'])}\n")
    print(f"Media angolo d'attacco: {np.mean(stats['start_angles'])}°")
    f.write(f"Media angolo d'attacco: {np.mean(stats['start_angles'])}°\n")
    print(f"Varianza angolo d'attacco: {np.var(stats['start_angles'])}")
    f.write(f"Varianza angolo d'attacco: {np.var(stats['start_angles'])}\n")
    print(f"Media framerate: {np.mean(stats['framerate'])}")
    f.write(f"Media framerate: {np.mean(stats['framerate'])}\n")
    print(f"Varianza framerate: {np.var(stats['framerate'])}")
    f.write(f"Varianza framerate: {np.mean(stats['framerate'])}\n")


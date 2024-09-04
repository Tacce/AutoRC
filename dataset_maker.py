from pathlib import Path
import numpy as np

from sequence import Sequence

sequences = []

for rosbag_path in list(Path('rosbags').iterdir()):
    sequences.append(Sequence(rosbag_path))

delta_f = 4
delta_p = 2
framerate = 10
max_velocity = 3

samples = []

for s in sequences:
    for t in range(s.num_frames):
        sample = s.get_sample(t, delta_f, delta_p, framerate, max_velocity)
        if sample is not None:
            samples.append(sample)

output_dir = 'samples_data'
Path(output_dir).mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(samples):
    np.savez(f'{output_dir}/sample_{i}.npz',
             point_cloud=np.asarray(sample.point_cloud.points),
             future_trajectory=sample.future_trajectory.points,
             past_trajectory=sample.past_trajectory.points,
             rgb_frame=sample.rgb_frame,
             depth_frame=sample.depth_frame,)

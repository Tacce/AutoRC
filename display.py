import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sequence import Sequence


rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_04_19-15_29_03"
delta_f = 4
delta_p = 2
framerate = 10
max_velocity = 3

s = Sequence(rosbag_path)
print(s.calcuate_start_angle())
print(s.low_quality)
print(s.lenght)
print(s.num_frames)
s.display_sequence()

for t in range(s.num_frames):
    sample = s.get_sample(min(t, s.num_frames - 1), delta_f, delta_p, framerate, max_velocity)
    if sample is not None:
        print(sample.calculate_max_distance_from_origin())
        print(sample.classify_trajectory())
        print(sample.get_trajectory_length())
        print(sample.framerate)
        print(sample.calculate_curve_angle())
        print(len(sample.future_trajectory.points))
        print(len(sample.past_trajectory.points))

        '''frame = np.rot90(sample.rgb_frame, 2) if s.is_rotated else sample.rgb_frame
        plt.imshow(frame)
        plt.axis('off')
        plt.show()'''

        sample.display()

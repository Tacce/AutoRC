import sys

import matplotlib.pyplot as plt
import numpy as np

from sequence import Sequence

rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_04_17-17_24_07\\"
t = 0
delta_f = 10
delta_p = 0

s = Sequence(rosbag_path)
print(s.calcuate_start_angle())
s.display_sequence()

sample = s.get_sample(min(t, s.num_frames - 1), delta_f, delta_p, 1, 3)
if sample is not None:
    print(s.lenght)
    print(sample.calculate_max_distance_from_origin())
    print(sample.classify_trajectory())
    print(sample.get_trajectory_length())
    print(sample.framerate)
    print(sample.calculate_curve_angle())
    print(len(sample.future_trajectory.points))

    '''frame = np.rot90(sample.rgb_frame, 2) if s.is_rotated else sample.rgb_frame
    plt.imshow(frame)
    plt.axis('off')
    plt.show()'''

    sample.display()
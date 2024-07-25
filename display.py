import sys

import matplotlib.pyplot as plt
import numpy as np

from sequence import Sequence

rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_05_31-16_50_46_ROTATED\\"
t = 10
delta_f = 12
delta_p = 3

s = Sequence(rosbag_path)
s.display_sequence()
sample = s.get_sample(min(t, s.lenght), delta_f, delta_p)

print(s.lenght)
print(sample.calculate_max_distance_from_origin())
print(sample.classify_trajectory())
print(sample.get_trajectory_length())
print(sample.framerate)

'''frame = np.rot90(sample.rgb_frame, 2) if s.is_rotated else sample.rgb_frame
plt.imshow(frame)
plt.axis('off')
plt.show()'''

sample.display()


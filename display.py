import sys
from sequence import Sequence

rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_05_31-16_50_16_ROTATED\\"
t = 10
delta_f = 10
delta_p = 5

s = Sequence(rosbag_path)
sample = s.get_sample(min(t, s.lenght - 1), delta_f, delta_p)
print(sample.classify_trajectory())
print(sample.get_trajectory_length())
sample.display()

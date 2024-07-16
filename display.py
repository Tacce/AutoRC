import sys
from sequence import Sequence

# rosbag_path = sys.argv[1]
rosbag_path = "rosbags/rosbag2_2024_05_31-16_50_16_ROTATED\\"
t = 6
delta_f = 4
delta_p = 0

s = Sequence(rosbag_path)
print(s.lenght)
sample = s.get_sample(min(t, s.lenght), delta_f, delta_p)
print(sample.classify_trajectory())
print(sample.get_trajectory_length())
sample.display()

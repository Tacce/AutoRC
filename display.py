import sys
from sequence import Sequence

rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_05_03-15_48_01\\"
t = 18
delta_f = 10
delta_p = 5

s = Sequence(rosbag_path)
sample = s.get_sample(min(t, s.lenght-1), delta_f, delta_p)
print(sample.classify_trajectory())
sample.display()

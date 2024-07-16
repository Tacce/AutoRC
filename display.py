import sys
from sequence import Sequence

rosbag_path = sys.argv[1]
# rosbag_path = "rosbags/rosbag2_2024_04_19-15_30_10\\"
t = 5
delta_f = 5
delta_p = 3

s = Sequence(rosbag_path)
print(s.lenght)
sample = s.get_sample(min(t, s.lenght), delta_f, delta_p)
print(sample.calculate_max_distance_from_origin())
print(sample.classify_trajectory())
print(sample.get_trajectory_length())
sample.display()

'''for t in range(0, int(s.lenght - delta_f), 2):
    sample = s.get_sample(min(t, s.lenght), delta_f, delta_p)
    print(str(t) + ": " + str(sample.calculate_max_distance_from_origin()))'''
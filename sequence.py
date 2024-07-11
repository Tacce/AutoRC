import sys
from pathlib import Path
import open3d as o3d
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R
from rosbags.image import message_to_cvimage

alignment_rotation = R.from_euler('xyz', [np.pi / 2, np.pi, np.pi / 2], degrees=False).as_matrix()

rotation_matrix_180 = np.array([[-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])


class Image:
    def __init__(self, timestamp, img):
        self.timestamp = timestamp
        self.img = img


class Sequence:
    def __init__(self, rosbag_path, is_rotated=False):

        with AnyReader([Path(rosbag_path)]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/zed/zed_node/depth/camera_info':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    k = msg.k
                    break

        self.intrinsic_matrix = np.array([[k[0], 0, k[2]],
                                          [0, k[4], k[5]],
                                          [0, 0, 1]])
        self.trajectory_data = []
        self.rgb_imgs = []
        self.depth_imgs = []

        with AnyReader([Path(rosbag_path)]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/zed/zed_node/odom':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    position = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
                    orientation = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                   msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
                    self.trajectory_data.append({'timestamp': timestamp, 'position': position,
                                                 'orientation': orientation})
                if connection.topic == '/zed/zed_node/right/image_rect_color':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, 'bgr8')
                    self.rgb_imgs.append(Image(timestamp, img))
                if connection.topic == '/zed/zed_node/depth/depth_registered':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, '32FC1')
                    self.depth_imgs.append(Image(timestamp, img))

        self.lenght = len(self.trajectory_data)

        self.is_rotated = is_rotated
        '''
        height, width = self.rgb_imgs[0].img.shape[:2]
        if height > 200 and width > 400:
            self.is_rotated = False
        else:
            self.is_rotated = True
        '''

    def __create_point_cloud_from_depth(self, depth_image, rgb_image, intrinsic_matrix):
        height, width = depth_image.shape
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        points = []
        colors = []

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                if z > 0:  # Ignora i punti senza profondità
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append((x, y, z))
                    colors.append(rgb_image[v, u] / 255.0)

        points = np.array(points)
        colors = np.array(colors)

        # Crea la nuvola di punti Open3D
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud

    def __transform_trajectory(self, trajectory, rotation):
        # Trasla la traiettoria in modo che il primo punto sia all'origine
        translation_vector = np.array(trajectory[0])
        translated_trajectory = trajectory - translation_vector

        # Applica la rotazione inversa
        inverse_rotation = np.linalg.inv(rotation)
        rotated_trajectory = translated_trajectory @ inverse_rotation.T

        return rotated_trajectory

    def get_sample(self, t, delta_f, delta_p):
        main_frame = self.trajectory_data[t]

        orientation = main_frame['orientation']
        timestamp = main_frame['timestamp']

        rotation = R.from_quat(orientation).as_matrix()
        rotation = rotation @ alignment_rotation

        closest_depth_frame = min(self.depth_imgs, key=lambda x: abs(x.timestamp - timestamp))
        closest_rgb_frame = min(self.rgb_imgs, key=lambda x: abs(x.timestamp - timestamp))

        closest_depth_frame.img = np.nan_to_num(closest_depth_frame.img, nan=0.0, posinf=0.0, neginf=0.0)

        point_cloud = self.__create_point_cloud_from_depth(closest_depth_frame.img, closest_rgb_frame.img,
                                                           self.intrinsic_matrix)

        trajectory_points_future = np.array([data['position'] for data in self.trajectory_data[t:t + delta_f]])
        trajectory_points_future = self.__transform_trajectory(trajectory_points_future, rotation)

        trajectory_points_past = np.array(
            [data['position'] for data in self.trajectory_data[max(0, t - delta_p):t + 1]])[::-1]
        trajectory_points_past = self.__transform_trajectory(trajectory_points_past, rotation)

        if self.is_rotated:
            trajectory_points_future = trajectory_points_future @ rotation_matrix_180.T
            trajectory_points_past = trajectory_points_past @ rotation_matrix_180.T
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) @ rotation_matrix_180.T)

        return point_cloud, trajectory_points_future, trajectory_points_past

    def display_sample(self, point_cloud, trajectory_points_future, trajectory_points_past):
        line_set_f = o3d.geometry.LineSet()
        line_set_f.points = o3d.utility.Vector3dVector(trajectory_points_future)
        lines = [[i, i + 1] for i in range(len(trajectory_points_future) - 1)]
        line_set_f.lines = o3d.utility.Vector2iVector(lines)

        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set_f.colors = o3d.utility.Vector3dVector(colors)

        line_set_p = o3d.geometry.LineSet()
        line_set_p.points = o3d.utility.Vector3dVector(trajectory_points_past)
        lines = [[i, i + 1] for i in range(len(trajectory_points_past) - 1)]
        line_set_p.lines = o3d.utility.Vector2iVector(lines)

        colors = [[0, 1, 0] for _ in range(len(lines))]
        line_set_p.colors = o3d.utility.Vector3dVector(colors)

        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=trajectory_points[0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # Visualizza le nuvole di punti trasformate e la traiettoria
        o3d.visualization.draw_geometries([point_cloud] + [line_set_f] + [line_set_p] + [coordinate_frame])


rosbag_path = sys.argv[1]
# rosbag_path = "rosbag2_2024_05_31-16_52_01\\"

# Istante da prendere in considerazione
t = 10
# Numero di punti della traiettoria passata e futura che si vogliono visualizzare
delta_f = 50
delta_p = 20
# Da impostare su True per le sequenze con la camera invertita
is_rotated = True

s = Sequence(rosbag_path, is_rotated)
s.display_sample(*s.get_sample(min(t, s.lenght-1), delta_f, delta_p))

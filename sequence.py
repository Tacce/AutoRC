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

angle_threshold = np.pi / 16
max_slope = 6


class Image:
    def __init__(self, timestamp, img):
        self.timestamp = timestamp
        self.img = img


class Sequence:
    def __init__(self, rosbag_path):

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

        self.lenght = (self.trajectory_data[-1]['timestamp'] - self.trajectory_data[0]['timestamp'])/1000000000

        if 'ROTATED' in str(rosbag_path):
            self.is_rotated = True
        else:
            self.is_rotated = False

        '''height, width = self.rgb_imgs[0].img.shape[:2]
        if height > 200 and width > 400:
            self.low_quality = False
        else:
            self.low_quality = True'''

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
        target_time = self.trajectory_data[0]['timestamp'] + t * 1000000000
        t = min(
            (i for i in range(len(self.trajectory_data)) if self.trajectory_data[i]['timestamp'] - target_time >= 0),
            key=lambda i: self.trajectory_data[i]['timestamp'] - target_time)

        main_frame = self.trajectory_data[t]

        delta_f *= 1000000000
        delta_p *= 1000000000

        orientation = main_frame['orientation']
        timestamp = main_frame['timestamp']

        rotation = R.from_quat(orientation).as_matrix()
        rotation = rotation @ alignment_rotation

        closest_depth_frame = min(self.depth_imgs, key=lambda x: abs(x.timestamp - timestamp))
        closest_rgb_frame = min(self.rgb_imgs, key=lambda x: abs(x.timestamp - timestamp))

        closest_depth_frame.img = np.nan_to_num(closest_depth_frame.img, nan=0.0, posinf=0.0, neginf=0.0)

        point_cloud = self.__create_point_cloud_from_depth(closest_depth_frame.img, closest_rgb_frame.img,
                                                           self.intrinsic_matrix)

        trajectory_points_future = np.array([data['position'] for data in self.trajectory_data[t:]
                                             if data['timestamp'] <= main_frame['timestamp'] + delta_f])
        trajectory_points_future = self.__transform_trajectory(trajectory_points_future, rotation)

        trajectory_points_past = np.array(
            [data['position'] for data in self.trajectory_data[0: t + 1]
             if data['timestamp'] >= main_frame['timestamp'] - delta_p])[::-1]
        trajectory_points_past = self.__transform_trajectory(trajectory_points_past, rotation)

        if self.is_rotated:
            trajectory_points_future = trajectory_points_future @ rotation_matrix_180.T
            trajectory_points_past = trajectory_points_past @ rotation_matrix_180.T
            point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) @ rotation_matrix_180.T)

        sample = Sample(point_cloud, trajectory_points_future, trajectory_points_past, closest_rgb_frame.img,
                        closest_depth_frame.img)

        return sample


class Sample:

    def __init__(self, point_cloud, trajectory_points_future, trajectory_points_past, rgb_frame, depth_frame):
        self.point_cloud = point_cloud
        self.future_trajectory = Trajectory(trajectory_points_future)
        self.past_trajectory = Trajectory(trajectory_points_past)
        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame

    def display(self):
        line_set_f = o3d.geometry.LineSet()
        line_set_f.points = o3d.utility.Vector3dVector(self.future_trajectory.points)
        lines = [[i, i + 1] for i in range(len(self.future_trajectory.points) - 1)]
        line_set_f.lines = o3d.utility.Vector2iVector(lines)

        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set_f.colors = o3d.utility.Vector3dVector(colors)

        line_set_p = o3d.geometry.LineSet()
        line_set_p.points = o3d.utility.Vector3dVector(self.past_trajectory.points)
        lines = [[i, i + 1] for i in range(len(self.past_trajectory.points) - 1)]
        line_set_p.lines = o3d.utility.Vector2iVector(lines)

        colors = [[0, 1, 0] for _ in range(len(lines))]
        line_set_p.colors = o3d.utility.Vector3dVector(colors)

        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=trajectory_points[0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # Visualizza le nuvole di punti trasformate e la traiettoria
        o3d.visualization.draw_geometries([self.point_cloud] + [line_set_f] + [line_set_p] + [coordinate_frame])

    def get_trajectory_length(self):
        return self.future_trajectory.length

    def classify_trajectory(self):
        return self.future_trajectory.classify()

    def calculate_covered_area_percentage(self):
        height, width = self.rgb_frame.shape[:2]
        total_pixels = width * height
        points = np.asarray(self.point_cloud.points)
        covered_pixels = set()

        for point in points:
            x, y, z = point
            if z > 0:  # Consider only points with positive depth
                # Project the 3D point to the 2D image plane (assuming pinhole camera model)
                u_proj = int((x / z) * width / 2 + width / 2)
                v_proj = int((y / z) * height / 2 + height / 2)
                # Check if the projected points are within the image boundaries
                if 0 <= u_proj < width and 0 <= v_proj < height:
                    covered_pixels.add((u_proj, v_proj))

        covered_area = len(covered_pixels)
        covered_area_percentage = (covered_area / total_pixels) * 100

        return covered_area_percentage


class Trajectory:
    def __init__(self, points):
        self.points = points
        self.length = np.sum(np.linalg.norm(self.points[1:] - self.points[:-1], axis=1))

    def classify(self):
        # Considero solo la traiettoria sul piano xz
        points = self.points[:, [0, 2]]

        diffs = np.diff(points, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_diffs = np.diff(angles)

        # Normalizza gli angoli in [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

        total_angle = np.sum(angle_diffs)

        slope = (points[-1, 1] - points[0, 1]) / (points[-1, 0] - points[0, 0]) if points[-1, 0] != points[
            0, 0] else np.inf

        if abs(slope) > max_slope:
            return 'straight'

        if np.abs(total_angle) < angle_threshold:
            if slope > 2:
                return 'straight'
            elif slope > 0:
                return 'right'
            else:
                return 'left'
        elif total_angle > 0:
            return 'left'
        else:
            return 'right'

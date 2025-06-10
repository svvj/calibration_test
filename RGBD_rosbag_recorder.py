#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from ouster.sdk import open_source
from ouster.sdk.client import LidarScan
import cv2
import numpy as np
import os
import subprocess

class LidarAndImagePublisher(Node):
    def __init__(self):
        super().__init__('lidar_and_image_publisher')

        # Publishers
        self.lidar_pub = self.create_publisher(PointCloud2, '/points', 10)
        self.image_pub = self.create_publisher(Image, '/images', 10)
        self.bridge = CvBridge()

        # Ouster source
        self.source = iter(open_source('os-122402000192.local'))  # Update with your sensor IP
        self.lidar_timer = self.create_timer(0.1, self.publish_lidar_scan)  # 10 Hz

        # Image source
        self.image_folder = '/path/to/your/frames'  # <--- CHANGE THIS
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png'))])
        self.image_index = 0
        self.image_timer = self.create_timer(0.1, self.publish_image_frame)

        # Rosbag record
        self.start_rosbag_record()

    def start_rosbag_record(self):
        try:
            self.rosbag_process = subprocess.Popen(
                ['ros2', 'bag', 'record', '/points', '/images'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info('Started rosbag recording for /points and /images')
        except Exception as e:
            self.get_logger().error(f'Failed to start rosbag recording: {e}')

    def publish_lidar_scan(self):
        try:
            scan = next(self.source)
            if isinstance(scan, LidarScan):
                range_data = scan.field('RANGE').astype(float)
                height, width = range_data.shape

                azimuth_angles = np.linspace(-np.pi, np.pi, width)
                elevation_angles = np.linspace(-np.pi / 4, np.pi / 4, height)
                azimuth_grid, elevation_grid = np.meshgrid(azimuth_angles, elevation_angles)

                x = range_data * np.cos(elevation_grid) * np.sin(azimuth_grid)
                y = range_data * np.cos(elevation_grid) * np.cos(azimuth_grid)
                z = range_data * np.sin(elevation_grid)

                xyz_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
                intensity = np.ones((xyz_points.shape[0], 1), dtype=np.float32)
                points = np.hstack((xyz_points, intensity))

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'ouster_frame'

                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
                ]

                msg = pc2.create_cloud(header, fields, points)
                self.lidar_pub.publish(msg)
                self.get_logger().info('Published LiDAR scan')

        except StopIteration:
            self.get_logger().info('LiDAR scan finished')
            self.shutdown()

    def publish_image_frame(self):
        if self.image_index >= len(self.image_files):
            self.get_logger().info('All image frames published')
            return

        img_path = os.path.join(self.image_folder, self.image_files[self.image_index])
        img = cv2.imread(img_path)

        if img is None:
            self.get_logger().warning(f"Failed to read image: {img_path}")
            self.image_index += 1
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        self.image_pub.publish(msg)
        self.get_logger().info(f'Published image: {self.image_files[self.image_index]}')
        self.image_index += 1

    def shutdown(self):
        if hasattr(self, 'rosbag_process') and self.rosbag_process:
            self.rosbag_process.terminate()
            self.rosbag_process.wait()
            self.get_logger().info('Stopped rosbag recording')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = LidarAndImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down from KeyboardInterrupt...')
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

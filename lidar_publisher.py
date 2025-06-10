import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from ouster.sdk import open_source
from ouster.sdk.client import LidarScan
import numpy as np
import subprocess

class LidarScanPublisher(Node):
    def __init__(self):
        super().__init__('lidar_scan_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/points', 10)
        self.source = iter(open_source('os-992127000062.local'))  # Replace with your sensor's hostname or IP
        self.source_iter = iter(self.source)
        self.timer = self.create_timer(0.1, self.publish_scan)  # Publish at 10 Hz

        # Start recording the rosbag
        self.start_rosbag_record()

    def start_rosbag_record(self):
        try:
            self.rosbag_process = subprocess.Popen(
                ['ros2', 'bag', 'record', '/points'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info('Started recording rosbag for /points')
        except Exception as e:
            self.get_logger().error(f'Failed to start rosbag recording: {e}')

    def publish_scan(self):
        try:
            scan = next(self.source)
            if isinstance(scan, LidarScan):
                # Extract and process RANGE data
                range_data = scan.field('RANGE').astype(float)
                height, width = range_data.shape

                # Generate simple XYZ coordinates based on the Lidar's geometry
                azimuth_angles = np.linspace(-np.pi, np.pi, width)
                elevation_angles = np.linspace(-np.pi / 4, np.pi / 4, height)
                azimuth_grid, elevation_grid = np.meshgrid(azimuth_angles, elevation_angles)

                x = range_data * np.cos(elevation_grid) * np.sin(azimuth_grid)
                y = range_data * np.cos(elevation_grid) * np.cos(azimuth_grid)
                z = range_data * np.sin(elevation_grid)

                xyz_points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

                # Create dummy intensity data (if needed)
                intensity = np.ones((xyz_points.shape[0], 1), dtype=np.float32)
                points = np.hstack((xyz_points, intensity))

                # Create header for PointCloud2 message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'ouster_frame'

                # Define PointCloud2 fields
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
                ]

                # Create and publish the PointCloud2 message
                pointcloud_msg = pc2.create_cloud(header, fields, points)
                self.publisher_.publish(pointcloud_msg)
                self.get_logger().info('Published original LidarScan as PointCloud2')


        except StopIteration:
            self.get_logger().info('No more data to process. Shutting down...')
            self.stop_rosbag_record()
            rclpy.shutdown()

    def stop_rosbag_record(self):
        if hasattr(self, 'rosbag_process') and self.rosbag_process:
            self.rosbag_process.terminate()
            self.rosbag_process.wait()
            self.get_logger().info('Stopped rosbag recording')

def main(args=None):
    rclpy.init(args=args)
    node = LidarScanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        node.stop_rosbag_record()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

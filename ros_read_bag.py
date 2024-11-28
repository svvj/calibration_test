import os
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs_py.point_cloud2 import read_points
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import open3d as o3d


def read_all_pointclouds_from_bag(bag_filename, points_topic, output_directory):
    """
    Reads all PointCloud2 messages from a ROS2 bag file and saves each frame as a .ply file.

    Args:
        bag_filename (str): Path to the ROS2 bag file.
        points_topic (str): Name of the PointCloud2 topic in the bag.
        output_directory (str): Directory to save .ply files.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Initialize ROS2 node
    rclpy.init()

    # Setup the bag reader
    storage_options = StorageOptions(uri=bag_filename, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Create and apply a filter for the point cloud topic
    from rosbag2_py import StorageFilter
    filter = StorageFilter()
    filter.topics = [points_topic]
    reader.set_filter(filter)

    # Read and save each point cloud frame
    frame_idx = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == points_topic:
            # Deserialize the message as PointCloud2
            from sensor_msgs.msg import PointCloud2
            pointcloud_msg = deserialize_message(data, PointCloud2)

            # Extract x, y, z points into a (N, 3) NumPy array
            points = np.array([[p[0], p[1], p[2]] for p in read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True)])

            # Save to a .ply file
            save_to_ply(points, output_directory, frame_idx)
            print(f"Saved frame {frame_idx} with {points.shape[0]} points.")
            frame_idx += 1

    rclpy.shutdown()  # Clean up ROS2 context


def save_to_ply(points, output_directory, frame_idx):
    """
    Save a numpy array of points to a .ply file using Open3D.

    Args:
        points (np.ndarray): Point cloud data as a (N, 3) NumPy array.
        output_directory (str): Directory to save the .ply file.
        frame_idx (int): Index of the current frame (used for naming files).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ply_filename = os.path.join(output_directory, f"frame_{frame_idx:04d}.ply")
    o3d.io.write_point_cloud(ply_filename, pcd)


if __name__ == "__main__":
    # Configuration
    bag_file = "/home/hci/workspace/sw/input/ouster/rosbag2_2024_11_11-14_55_01"  # Path to your ROS2 bag file
    pointcloud_topic = "/ouster/points"  # The PointCloud2 topic in your bag
    output_directory = "./output_pointclouds"  # Directory to save .ply files

    # Read all point clouds from the bag and save to .ply files
    read_all_pointclouds_from_bag(bag_file, pointcloud_topic, output_directory)

    print(f"All point clouds have been saved to {output_directory}")

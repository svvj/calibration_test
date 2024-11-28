import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_calibration(calib_file):
    """
    Load the calibration matrix from calib.json.
    """
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    values = calib_data["results"]["T_lidar_camera"]
    T_lidar_camera = np.eye(4)
    # Extract translation and quaternion
    translation = np.array(values[:3])  # [x, y, z]
    quaternion = np.array(values[3:])  # [qx, qy, qz, qw]

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]]).as_matrix()

    # Fill in the transformation matrix
    T_lidar_camera[:3, :3] = rotation_matrix
    T_lidar_camera[:3, 3] = translation

    return T_lidar_camera



def load_point_cloud(ply_file):
    """
    Load 3D point cloud from a .ply file.
    """
    pcd = o3d.io.read_point_cloud(ply_file)
    return np.asarray(pcd.points)


def transform_points(points, T_lidar_camera):
    """
    Transform points from LiDAR frame to Camera frame.
    """
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # Convert to homogeneous coordinates
    points_camera = (T_lidar_camera @ points_h.T).T
    return points_camera[:, :3]  # Remove the homogeneous coordinate


def project_points_equirectangular(points_camera, image_width, image_height):
    """
    Project 3D points from the camera frame onto an equirectangular image.
    """
    # print infos
    print("points_camera: ", points_camera)
    print("image_width: ", image_width)
    print("image_height: ", image_height)

    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute spherical coordinates
    phi = np.arctan2(y, x)  # [-pi, pi]
    theta = np.arcsin(z / r)  # [-pi/2, pi/2]

    # Map to pixel coordinates in the equirectangular image
    u = (phi / (2 * np.pi) + 0.5) * image_width
    v = (0.5 - theta / np.pi) * image_height

    return np.column_stack((u, v, r))


def visualize_projection(image_path, points_2d, image_width, image_height):
    """
    Visualize the 2D projection of 3D points onto the image.
    """
    # Load the equirectangular image
    image = plt.imread(image_path) if image_path else np.zeros((image_height, image_width))

    plt.figure(figsize=(12, 6))
    plt.imshow(image, extent=[0, image_width, image_height, 0])
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=0.1, c=points_2d[:, 2], alpha=0.7)
    plt.title("Projected Point Cloud on Equirectangular Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # File paths
    calib_file = "calibration_outputs/calib.json"
    ply_file = "calibration_outputs/rosbag2_2024_11_11-15_02_50_0.db3.ply"
    image_width = 2880  # Example width for equirectangular image
    image_height = 1440  # Example height for equirectangular image
    image_path = None  # Replace with the path to your equirectangular image

    # Load calibration and point cloud
    T_lidar_camera = load_calibration(calib_file)
    points_lidar = load_point_cloud(ply_file)

    # Transform points to the camera frame
    points_camera = transform_points(points_lidar, T_lidar_camera)

    # Project points to equirectangular image
    points_2d = project_points_equirectangular(points_camera, image_width, image_height)

    # Visualize the projection
    visualize_projection(image_path, points_2d, image_width, image_height)

import os
import cv2
import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def convert_and_save_images(input_image_path, output_image_path, grayscale_image_path):
    # Load the original image
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")
    
    # if the images are already there, skip the conversion
    if os.path.exists(output_image_path) and os.path.exists(grayscale_image_path):
        print(f"Images already exist. Skipping the conversion.")
        return

    # Save the image as image_000001.jpg
    cv2.imwrite(output_image_path, original_image)
    print(f"Image saved to {output_image_path}")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image
    cv2.imwrite(grayscale_image_path, gray_image)
    print(f"Grayscale image saved to {grayscale_image_path}")


def load_calib_json_data(json_path, print_values=False):
    calibration_result = json.load(open(json_path))

    # rosbag information
    bag_name = calibration_result["meta"]["bag_names"][0]
    # Extract the transformation matrix
    T_lidar_camera = calibration_result["results"]["T_lidar_camera"]
    # Convert the transformation matrix to the right format


    # Prepare the camera intrinsics
    camera_model = calibration_result["camera"]["camera_model"]
    camera_intrinsics = calibration_result["camera"]["intrinsics"]

    if print_values:
        print("Transformation matrix from LiDAR to camera:")
        print(T_lidar_camera)

        print("Camera model:", camera_model)
        print("Camera intrinsics:", camera_intrinsics)

    print("Calibration data loaded successfully.")

    # return the values
    return bag_name, T_lidar_camera, camera_model, camera_intrinsics


def load_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_np = np.asarray(pcd.points)
    print("Point cloud loaded successfully, shape:", pcd_np.shape)
    
    # save points to ply
    sparse_ply_path = ply_path.replace(".ply", "_sparse.ply")
    o3d.io.write_point_cloud(sparse_ply_path, pcd)
    print(f"Point cloud saved to {sparse_ply_path}")
    return pcd_np


def invert_transform_matrix(matrix):
    """Invert a 4x4 transformation matrix."""
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    inv_R = R.T
    inv_t = -np.dot(inv_R, t)
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = inv_R
    inv_matrix[:3, 3] = inv_t
    return inv_matrix


def quaternion_to_rotation_matrix(Q):
    # Q [qx qy qz qw]
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * Q[1]**2 - 2 * Q[2]**2
    R[0, 1] = 2 * Q[0] * Q[1] - 2 * Q[2] * Q[3]
    R[0, 2] = 2 * Q[0] * Q[2] + 2 * Q[1] * Q[3]
    R[1, 0] = 2 * Q[0] * Q[1] + 2 * Q[2] * Q[3]
    R[1, 1] = 1 - 2 * Q[0]**2 - 2 * Q[2]**2
    R[1, 2] = 2 * Q[1] * Q[2] - 2 * Q[0] * Q[3]
    R[2, 0] = 2 * Q[0] * Q[2] - 2 * Q[1] * Q[3]
    R[2, 1] = 2 * Q[1] * Q[2] + 2 * Q[0] * Q[3]
    R[2, 2] = 1 - 2 * Q[0]**2 - 2 * Q[1]**2
    return R


import numpy as np


def equirectangular_projection_batch(intrinsic, points_3d):
    """
    Projects multiple 3D points into equirectangular coordinates.

    Args:
        intrinsic (list or np.ndarray): Intrinsic parameters [width, height].
        points_3d (np.ndarray): An array of 3D points with shape (n, 3).

    Returns:
        np.ndarray: An array of 2D points in equirectangular projection with shape (n, 2).
    """
    # Ensure points_3d is a numpy array
    points_3d = np.asarray(points_3d)

    # Compute squared norms for all points
    squared_norms = np.sum(points_3d ** 2, axis=1)

    # Handle points with very small norms (near the origin)
    near_origin_mask = squared_norms < 1e-3
    default_coords = np.array([intrinsic[0] / 2, intrinsic[1] / 2])

    # Normalize points to calculate bearing
    bearings = points_3d / np.linalg.norm(points_3d, axis=1, keepdims=True)

    # Compute latitude and longitude
    lat = -np.arcsin(bearings[:, 1])  # y-component
    lon = np.arctan2(bearings[:, 0], bearings[:, 2])  # x/z components

    # Compute equirectangular coordinates
    x = intrinsic[0] * (0.5 + lon / (2.0 * np.pi))
    y = intrinsic[1] * (0.5 - lat / np.pi)

    # Combine x and y into final output
    projected_points = np.stack([x, y], axis=1)

    # Replace points near origin with default coordinates
    projected_points[near_origin_mask] = default_coords

    return projected_points


import numpy as np


def project_points_to_image(points, T_lidar_camera, image_shape):
    """
    Projects 3D LiDAR points onto the 2D image using the transformation matrix and equirectangular projection.
    Args:
        points: List of 3D points [x, y, z].
        T_lidar_camera: Transformation matrix from LiDAR to camera frame, TUM format.
        image_shape: Shape of the target image [width, height].
    Returns:
        List of 2D image points, normalized depths, and valid points in the image bounds.
    """
    print("Input T_lidar_camera: ", T_lidar_camera)

    # Prepare the transformation matrix
    T = np.array(T_lidar_camera)  # TUM [tx ty tz qx qy qz qw]
    R_quat_vec = T[3:]  # Quaternion [qx qy qz qw]
    t_vec = T[:3]  # Translation [tx ty tz]

    # Convert quaternion to rotation matrix
    R_mat = quaternion_to_rotation_matrix(R_quat_vec)
    T_l_2_c = np.eye(4)
    T_l_2_c[:3, :3] = R_mat
    T_l_2_c[:3, 3] = t_vec

    print("T_l_2_c: ", T_l_2_c)

    # Invert transformation matrix for camera-to-LiDAR transformation
    T_camera_lidar = invert_transform_matrix(T_l_2_c)
    print("T_camera_lidar: ", T_camera_lidar)

    # Prepare 3D points (add a 1 for homogeneous coordinates)
    points = np.array(points)
    points = np.hstack([points, np.ones((points.shape[0], 1))])

    # Transform points to the camera frame
    points_camera = (T_camera_lidar @ points.T).T[:, :3]

    # Filter points by distance
    points_camera = points_camera[points_camera[:, 2] < 100]

    # Use image_shape directly for equirectangular intrinsic
    intrinsic = image_shape  # width, height for equirectangular projection
    projected_points = equirectangular_projection_batch(intrinsic, points_camera)

    # Calculate depths for coloring
    depths = np.linalg.norm(points_camera, axis=1)

    # Normalize depths
    depth_min = depths.min()
    depth_max = depths.max()
    depths_normalized = (depths - depth_min) / (depth_max - depth_min)

    print("Depth min: ", depth_min, "Depth max: ", depth_max)

    # Filter valid points within the image bounds
    valid_indices = (
            (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_shape[0]) &  # Width check
            (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_shape[1])  # Height check
    )
    valid_points = points_camera[valid_indices]
    valid_projected_points = projected_points[valid_indices]
    valid_depths = depths_normalized[valid_indices]

    return valid_projected_points, valid_depths, valid_points


def visualize_projected_points(image, depths, points_2d, image_idx=-1):
    """
    Visualize the 2D projected points on the image.
    Args:
        image: The target image.
        points_2d: The 2D points to visualize on the image.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=0.3, c=depths, cmap='viridis', alpha=0.3)
    # plt.colorbar(label='Depth')
    plt.axis("off")
    if image_idx >= 0:
        plt.savefig(f'output/image_{image_idx}.png', bbox_inches='tight')
        print(f"Image saved as ./output/image_{image_idx}.png")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Prepare the images
    input_image_path = "calibration_outputs/original_image.png"
    output_image_path = "data/images/image_000001.jpg"
    grayscale_image_path = "data/sample_images/gray_image.jpg"
    
    convert_and_save_images(input_image_path, output_image_path, grayscale_image_path)

    # Load the images
    original_image = cv2.imread(output_image_path)
    gray_image = cv2.imread(grayscale_image_path)

    # Prepare the point cloud
    # ply_folder_path = "output_pointclouds"
    # ply_files_list = os.listdir(ply_folder_path)

    ply_folder_path = "calibration_outputs"
    ply_files_list = ["rosbag2_2024_11_11-14_55_01_0.db3.ply"]

    for i, ply_file in enumerate(ply_files_list):
        ply_path = os.path.join(ply_folder_path, ply_file)
            
        # ply_path = "output_pointclouds/frame_0000.ply"

        # Prepare the calibration result (Transform matrix)
        calibration_result_path = "calibration_outputs/calib.json"
        # read json file
        bag_name, T_lidar_camera, camera_model, camera_intrinsics = load_calib_json_data(calibration_result_path, False)

        points = load_ply(ply_path)

        # Visulalize point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])

        # Visualize registrated point cloud with grey image
        points_2d, depths, valid_points = project_points_to_image(points, np.array(T_lidar_camera), camera_intrinsics)
        if len(ply_files_list) > 1:
            visualize_projected_points(gray_image, depths, points_2d, i)
        else:
            visualize_projected_points(gray_image, depths, points_2d)
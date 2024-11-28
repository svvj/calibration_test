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
    pcd_np = np.asarray(pcd.points)
    print("Point cloud loaded successfully, shape:", pcd_np.shape)
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


def project_points_to_image(points, T_lidar_camera, image_shape):
    """
    Projects 3D LiDAR points onto the 2D image using the transformation matrix and camera intrinsics.
    Args:
        points: List of 3D points [x, y, z].
        T_lidar_camera: Transformation matrix from LiDAR to camera frame, TUM format.
        camera_intrinsics: Camera intrinsic parameters.
        image_shape: Shape of the target image (height, width).
    Returns:
        List of 2D image points and valid points in the image bounds with colored depth.
    """
    print("input T_lidar_camera: ", T_lidar_camera)

    # Prepare the transformation matrix
    T = np.array(T_lidar_camera)    # TUM [tx ty tz qx qy qz qw]
    R_quat_vec = T[3:]                       # Quaternion [qx qy qz qw]
    t_vec = T[:3]                       # Translation [tx ty tz]
    # We need to convert the quaternion to rotation matrix
    R_mat = quaternion_to_rotation_matrix(R_quat_vec)
    t = t_vec
    T_l_2_c = np.eye(4)
    T_l_2_c[:3, :3] = R_mat
    T_l_2_c[:3, 3] = t
    print("T_l_2_c: ", T_l_2_c)

    T_camera_lidar = invert_transform_matrix(T_l_2_c)
    # T_camera_lidar = T_l_2_c
    print("T_camera_lidar: ", T_camera_lidar)

    points = np.array(points)
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    # points = np.dot(T_camera_lidar, points.T).T

    points = points[:, :3].astype(np.float64)

    # if the points are too far, delete them
    points = points[points[:, 2] < 100]

    # visualize the transformed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axis])

    
    # Equirectangular projection
    points_2d = np.zeros((points.shape[0], 2))
    valid_points = []
    depths = []
    for i, point in enumerate(points):
        # calculate the projection
        x, y, z = point[0], point[1], point[2]
        depth = np.sqrt(x**2 + y**2 + z**2)

        phi = np.arctan2(x, y) # Longitude, [-pi, pi]
        theta = np.arcsin(z / depth) # Latitude, [-pi/2, pi/2]

        # Map spherical coordinates to pixel coordinates (opencv)
        u = (phi / (2 * np.pi) + 0.5) * image_shape[0]
        v = (0.5 - theta / np.pi) * image_shape[1]
        
        # # Combine into pixel coordinates
        # points_2d[i] = [u, v]

        # distance from the camera
        depth = np.linalg.norm(point)
        if depth > 100:
            depths.append(-1)
            continue

        # colorize the depth
        depths.append(depth)

        # Check if the point is within the image bounds
        if depths[-1] == -1:
            points_2d[i] = [-1, -1]
        else:
            points_2d[i] = [u, v]
        # points_2d[i] = [x, y]
        valid_points.append(point)
        
    depths = np.array(depths)
    print("Depth min: ", depths.min(), "Depth max: ", depths.max())
    depth_range = depths.max() - depths.min()
    depth_min = depths.min()
    depths = (depths - depth_min) / depth_range
    # # plot depth distribution
    # plt.hist(depths, bins=100)
    # plt.show()

    return points_2d, depths, valid_points

def visualize_projected_points(image, depths, points_2d):
    """
    Visualize the 2D projected points on the image.
    Args:
        image: The target image.
        points_2d: The 2D points to visualize on the image.
    """
    plt.figure(figsize=(14, 7))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # white background
    white_image = np.ones_like(image) * 255
    plt.imshow(white_image)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=0.1, c=depths, cmap='viridis', alpha=0.3)
    # plt.colorbar(label='Depth')
    # plt.title("Projected 3D Points onto Image")
    plt.axis("off")
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
    ply_path = "calibration_outputs/rosbag2_2024_11_11-14_55_01_0.db3.ply"

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
    visualize_projected_points(gray_image, depths, points_2d)
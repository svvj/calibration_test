import os
import open3d as o3d

def visualize_ply(file_path):
    # Read the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Path to the PLY file
    ply_folder_path = "output_pointclouds"
    ply_file_list = os.listdir(ply_folder_path)
    print("Find ", len(ply_file_list), " files in the folder")
    ply_idx = 0

    ply_path = os.path.join(ply_folder_path, ply_file_list[ply_idx])
    print(ply_path)
    
    # Visualize the PLY file
    visualize_ply(ply_path)
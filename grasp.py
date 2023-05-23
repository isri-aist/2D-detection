import open3d as o3d
import numpy as np

color_raw = o3d.io.read_image("rgb_image.jpg")
depth_raw = o3d.io.read_image("depth_image.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
width = 640
height = 480
fx = 528.2193663864483
fy = 524.1409341228491
cx = 320.1900305884105
cy = 225.9266226598864
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))

def ray_casting(point_cloud, origin, direction, distance_threshold):
    # Create a KD-Tree for efficient ray casting
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    # Calculate ray end points
    end_point = origin + direction * distance_threshold

    # Perform ray casting
    [_, idx, _] = kdtree.search_radius_vector_3d(origin, distance_threshold)
    [_, idx2, _] = kdtree.search_radius_vector_3d(end_point, distance_threshold)

    # Check if any points are encountered along the ray
    encountered_points = np.intersect1d(idx, idx2)

    if len(encountered_points) > 0:
        print("Ray encountered points.")
    else:
        print("Ray is free to travel.")

# Define the ray parameters
#    x: 0.21586781040909342
    # y: 0.18640155456223595
    # z: 0.763

object = np.array([0.21586781040909342, 0.18640155456223595, 0.763])  # Ray origin
origin = np.array([0, 0, 0])  # Ray direction
distance_threshold = 1.0  # Maximum distance to check

# Perform ray casting
ray_casting(pcd, origin, object, distance_threshold)

#show the ray between origin and object in red
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(np.vstack((origin, object)))
line_set.lines = o3d.utility.Vector2iVector([[0, 1]])


grasping_postion = object + np.array([0, 0, -0.15])

#show the ray between origin and grapsing position in red
line_set2 = o3d.geometry.LineSet()
line_set2.points = o3d.utility.Vector3dVector(np.vstack((origin, grasping_postion)))
line_set2.lines = o3d.utility.Vector2iVector([[0, 1]])


#show x,y,z axis
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

#orientation
#   orientation: 
#     x: 0.0
#     y: 0.7071067811865475
#     z: 0.0
#     w: 0.7071067811865476

quaternion = np.array([0.7071067811865476, 0.0, 0.7071067811865475, 0.0])
rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
z_axis = rotation_matrix[:3, 2]

end = object + z_axis * 0.02

#show the ray between object and end in red
line_set3 = o3d.geometry.LineSet()
line_set3.points = o3d.utility.Vector3dVector(np.vstack((object, end)))
line_set3.lines = o3d.utility.Vector2iVector([[0, 1]])

# Normalize the z-axis vector
z_axis_normalized = z_axis / np.linalg.norm(z_axis)

# Generate points on the circle
theta = np.linspace(0, 2*np.pi, num=100)  # Adjust the 'num' parameter to increase/decrease the number of points
circle_points = 0.15 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))

# Transform the circle points to match the orientation of the z-axis
transformed_circle_points = circle_points @ rotation_matrix[:3, :3].T

# Transform the circle points to be around end
transformed_circle_points = transformed_circle_points + end

# Create a PointCloud object with the transformed circle points
circle_cloud = o3d.geometry.PointCloud()
circle_cloud.points = o3d.utility.Vector3dVector(transformed_circle_points)

#select the vertical point of the circle
highest_point_index = np.argmin(transformed_circle_points[:, 1])
highest_point = transformed_circle_points[highest_point_index]

#plot the point in blue
line_set4 = o3d.geometry.LineSet()
line_set4.points = o3d.utility.Vector3dVector(np.vstack((end, highest_point)))
line_set4.lines = o3d.utility.Vector2iVector([[0, 1]])


#direct computation of highest point
highest_point2 = np.array([0, -1, 0]) * 0.15 
highest_point2 = highest_point2 @ rotation_matrix[:3, :3].T
highest_point2 = highest_point2 + end

#plot the point in blue
line_set5 = o3d.geometry.LineSet()
line_set5.points = o3d.utility.Vector3dVector(np.vstack((end, highest_point2)))
line_set5.lines = o3d.utility.Vector2iVector([[0, 1]])

print("object", object)
print("highest_point2", highest_point2)


o3d.visualization.draw_geometries([pcd, mesh_frame, line_set, line_set2, line_set3, circle_cloud, line_set4, line_set5])

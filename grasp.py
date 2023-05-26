import open3d as o3d
import numpy as np
import scipy

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
# 0.254599 -0.116096  0.629787

object = np.array([0.254599, -0.116096, 0.629787])  # Ray origin
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
#quaternion: 0.654509 0.444547 0.172828 0.586624

quaternion = np.array([0.586624, 0.654509, 0.444547, 0.172828])
rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
x_axis = rotation_matrix[:3, 0]
y_axis = rotation_matrix[:3, 1]
z_axis = rotation_matrix[:3, 2]


end = object + z_axis * 0.02

#show the ray between object and end in red
line_set3 = o3d.geometry.LineSet()
line_set3.points = o3d.utility.Vector3dVector(np.vstack((object, end)))
line_set3.lines = o3d.utility.Vector2iVector([[0, 1]])

# Normalize the z-axis vector
z_axis_normalized = z_axis / np.linalg.norm(z_axis)

# Generate points on the circle
# theta = np.linspace(0, 2*np.pi, num=100)  # Adjust the 'num' parameter to increase/decrease the number of points
# circle_points = 0.15 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))

# # Transform the circle points to match the orientation of the z-axis
# transformed_circle_points = circle_points @ rotation_matrix[:3, :3].T

def circle_vector(t):
    return np.cos(t) *x_axis + np.sin(t) * y_axis

transformed_circle_points = np.array([circle_vector(t) for t in np.linspace(0, 2*np.pi, num=100)])*0.15

# Transform the circle points to be around end
transformed_circle_points = transformed_circle_points + end

# Create a PointCloud object with the transformed circle points
circle_cloud = o3d.geometry.PointCloud()
circle_cloud.points = o3d.utility.Vector3dVector(transformed_circle_points)

#[info] [wdot] camera Position:  0.164297 0.0251142   1.08364 quaternion: -0.605256  0.588918 -0.375284  0.382104

camera_position = np.array([0.164297, 0.0251142, 1.08364])
camera_quaternion = np.array([0.382104, -0.605256, 0.588918, -0.375284])
camera_rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(camera_quaternion)
print("camera_rotation_matrix", camera_rotation_matrix)
print("camera_rotation_matrix inverse", np.linalg.inv(camera_rotation_matrix))


base_circle_points = transformed_circle_points @ camera_rotation_matrix[:3, :3].T

#select highest point in base frame
highest_point_index = np.argmax(base_circle_points[:, 2])

line_set4 = o3d.geometry.LineSet()
line_set4.points = o3d.utility.Vector3dVector(np.vstack((end, transformed_circle_points[highest_point_index])))
line_set4.lines = o3d.utility.Vector2iVector([[0, 1]])


#direct computation of the highest point
up_vector_base = np.array([0, 0, 1])
up_vector = up_vector_base @ np.linalg.inv(camera_rotation_matrix[:3, :3].T)

horizontal_vector_base = np.array([-1, 0, 0])
horizontal_vector = horizontal_vector_base @ np.linalg.inv(camera_rotation_matrix[:3, :3].T)

print("up_vector", up_vector)

def verticality_of_circle_vector(t):
    return np.dot(circle_vector(t), up_vector)

def horizontal_of_circle_vector(t):
    return np.dot(circle_vector(t), horizontal_vector)

#find t such that verticality_of_circle_vector(t) is maximized
t = scipy.optimize.fmin(lambda t: -verticality_of_circle_vector(t), 0)[0]
#between 0 and 2pi
t = t % (2*np.pi)

print("t", t)

x_axis_in_base_frame = x_axis @ camera_rotation_matrix[:3, :3].T
y_axis_in_base_frame = y_axis @ camera_rotation_matrix[:3, :3].T

print("x_axis_in_base_frame", x_axis_in_base_frame)
print("y_axis_in_base_frame", y_axis_in_base_frame)

tbis = np.arctan2(y_axis_in_base_frame[0], x_axis_in_base_frame[0])
#check if tbis is the solution to minimize or maximize verticality_of_circle_vector
if horizontal_of_circle_vector(tbis) < horizontal_of_circle_vector(tbis + np.pi):
    tbis = tbis + np.pi

tbis = tbis % (2*np.pi)

print("tbis", tbis)

highest_point = circle_vector(t) * 0.15 + end
print("highest_point", highest_point)
print("dot product", verticality_of_circle_vector(t))

highest_point2 = circle_vector(tbis) * 0.15 + end
print("highest_point2", highest_point2)
print("dot product", verticality_of_circle_vector(tbis))

import matplotlib.pyplot as plt
plt.plot(np.linspace(0, 2*np.pi, num=100), [verticality_of_circle_vector(x) for x in np.linspace(0, 2*np.pi, num=100)])
#put vertical line at t, tbis
plt.axvline(t, color='r')
plt.axvline(tbis, color='g')
plt.show()

# end_base = end @ camera_rotation_matrix[:3, :3].T
# highest_point_base = end_base + np.array([0, 0, 0.15])
# highest_point = highest_point_base @ np.linalg.inv(camera_rotation_matrix[:3, :3].T)

line_set5 = o3d.geometry.LineSet()
line_set5.points = o3d.utility.Vector3dVector(np.vstack((end, highest_point)))
line_set5.lines = o3d.utility.Vector2iVector([[0, 1]])

line_set6 = o3d.geometry.LineSet()
line_set6.points = o3d.utility.Vector3dVector(np.vstack((end, highest_point2)))
line_set6.lines = o3d.utility.Vector2iVector([[0, 1]])

print("object", object)
print("highest_point", highest_point)

o3d.visualization.draw_geometries([pcd, mesh_frame, line_set, line_set2, line_set3, circle_cloud, line_set4, line_set5, line_set6])

#o3d.visualization.draw_geometries([pcd, mesh_frame, line_set, line_set2, line_set3, circle_cloud, line_set4, line_set5])

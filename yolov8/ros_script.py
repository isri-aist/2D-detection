from ultralytics import YOLO
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, String, Float64
import numpy as np
import tf2_ros
import message_filters

classesDict = dict()
classesDict[0] = 'CupNoodles'
classesDict[1] = 'CurryCup'
classesDict[2] = 'SeaFoodCup'



rgb_image_topic = '/head_camera/rgb/image_rect_color'
depth_image_topic = '/head_camera/depth/image_raw'
camera_info_topic = '/head_camera/depth/camera_info'
# rgb_image_topic = '/rgb/image_raw'
# depth_image_topic = '/depth/image_raw'
# camera_info_topic = '/depth/camera_info'


output_topic_prefix = '/object_pose_initialization/'

bridge = CvBridge()
rgb_data = None
depth_data = None
camera_info = dict()
model = YOLO('trained_models/best.pt')
pub_dict = dict()
node = None

def rgb_callback(data):
    global rgb_data
    try:
        rgb_data = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    process_images(data.header.stamp)

def depth_callback(data):
    global depth_data
    try:
        depth_data = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print(e)

def sync_callback(rgb_msg, depth_msg, command_msg):
    global rgb_data, depth_data
    try:
        rgb_data = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    try:
        depth_data = bridge.imgmsg_to_cv2(depth_msg, "passthrough")
    except CvBridgeError as e:
        print(e)
    
    process_images(rgb_msg.header.stamp)


def camera_info_callback(data):
    global camera_info
    camera_info['fx'] = data.K[0]
    camera_info['fy'] = data.K[4]
    camera_info['cx'] = data.K[2]
    camera_info['cy'] = data.K[5]

def process_images(time):
    global rgb_data, depth_data, camera_info, pub_dict, pub_end, node
    
    if rgb_data is not None and depth_data is not None and camera_info:
        #time
        start = rclpy.Time.now()
        #inference
        results = model(rgb_data, stream=True).__next__().boxes

        #end time
        end = rclpy.Time.now()
        duration = end - start
        node.loginfo('Inference time (s): {}'.format(duration.to_sec()))
        pub_detection_time.publish(duration.to_sec())

        detected_objects_txt = ''

        #check if there is at least one object detected
        for i in range(len(results.boxes)):
            objectClass_id = int(results.cls[i].item())
            objectClass = classesDict[objectClass_id]
            #if not found in the dictionary, skip
            if objectClass is None:
                print('Warning: object class not found in the dictionary. Check that the model was trained with the same classes as the dictionary.')
                continue
            detected_objects_txt += objectClass + ' '
            confidence = results.conf[i].item()
            #if confidence is too low, skip
            if confidence < 0.3:
                print("Warning: Confidence too low, not keeping detected object from class", objectClass)
                continue
            xywh = results.xywh[i].cpu().numpy()
            x = int(xywh[0])
            y = int(xywh[1])
            w = xywh[2]
            h = xywh[3]

            z = depth_data[y][x]

            k_search = 0
            while z == 0 and k_search < min(w//2, h//2):
                k_search += 1
                for i in range(-k_search, k_search):
                    for j in range(-k_search, k_search):
                        try:
                            z = depth_data[y+i][x+j]
                        except:
                            z = 0
                        if z != 0:
                            break
                    if z != 0:
                        break

            # 3D coordinates of the object center
            X = (x - camera_info['cx']) * z / camera_info['fx']
            Y = (y - camera_info['cy']) * z / camera_info['fy']
            Z = z + 45 #object has 4.5 cm radius

            initial = np.array([0, 0, 1]) #default orientation of the object
            ratio = h / w
            print("ratio", ratio)
            if ratio > 1.09:
                direction = np.array([0, 1, 0])
            elif ratio < 0.91:
                direction = np.array([1, 0, 0])
            else:
                direction = np.array([0, 0, -1])

            #check if initial and direction are parallel and opposite
            if np.dot(initial, direction) == -1:
                #180 degrees rotation quaternion
                quat = np.array([0, 1, 0, 0])
            else:
                axis = np.cross(initial, direction)
                angle = np.arccos(np.dot(initial, direction) / (np.linalg.norm(initial) * np.linalg.norm(direction)))

                # Compute the rotation as a quaternion
                quat = tf.transformations.quaternion_about_axis(angle, axis)

            # publish the result to the output topic
            pose = PoseStamped()
            pose.header.frame_id = 'head_camera_rgb_optical_frame'
            pose.header.stamp = time
            pose.pose.position.x = X * 0.001
            pose.pose.position.y = Y * 0.001
            pose.pose.position.z = Z * 0.001

            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pub = pub_dict[objectClass_id]
            pub.publish(pose)
            node.loginfo('Object detected: {}'.format(objectClass))
            node.loginfo('Confidence: {}'.format(confidence))
            node.loginfo('Pose: {}'.format(pose))
        #send end of detection signal
        pub_end.publish(detected_objects_txt)
        #kill node
        #rclpy.signal_shutdown('All objects detected have been processed')


def listener():
    global pub_dict
    global node
    # initialize the node
    rclpy.init()
    node = rclpy.create_node('RGB_detection')

    #
    pub_end = node.create_publisher(String, "/end_of_detection", 1)
    pub_detection_time = node.create_publisher(Float64, "/detection_time", 1)

    # subscribe to the input topics
    sub_rgb = message_filters.Subscriber(node, Image, rgb_image_topic)
    sub_depth = message_filters.Subscriber(node, Image, depth_image_topic)
    sub_camera_info = node.create_subscription(CameraInfo, camera_info_topic, camera_info_callback, 1)
    sub_command = message_filters.Subscriber(node, Bool, "/start_detection_command")
    combined_sub = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth, sub_command], 1, 0.1, allow_headerless=True)

    combined_sub.registerCallback(sync_callback)

    # create a publisher for the output topic for each object class
    for objectClass_id in classesDict:
        objectClass = classesDict[objectClass_id]
        output_topic = output_topic_prefix + objectClass
        pub = node.create_publisher(PoseStamped, output_topic, 1)
        pub_dict[objectClass_id] = pub

    # spin 
    rclpy.spin(node)

    

if __name__ == '__main__':
    listener()


######TODO
# handle multiple objects classes with global list and multiple publishers
# maybe transform frame from rgb to depth camera frame
# maybe use depth image rectified

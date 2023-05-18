from ultralytics import YOLO
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import tf
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

def sync_callback(rgb_msg, depth_msg):
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
    global rgb_data, depth_data, camera_info, pub_dict
    
    if rgb_data is not None and depth_data is not None and camera_info:
        #inference
        results = model(rgb_data, stream=True).__next__().boxes

        #check if there is at least one object detected
        for i in range(len(results.boxes)):
            objectClass_id = int(results.cls[i].item())
            objectClass = classesDict[objectClass_id]
            #if not found in the dictionary, skip
            if objectClass is None:
                print('Warning: object class not found in the dictionary. Check that the model was trained with the same classes as the dictionary.')
                continue
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
            rospy.loginfo('Object detected: {}'.format(objectClass))
            rospy.loginfo('Confidence: {}'.format(confidence))
            rospy.loginfo('Pose: {}'.format(pose))
            #wait 1s
            rospy.sleep(5)
        #kill node
        rospy.signal_shutdown('All objects detected have been processed')


def listener():
    global pub_dict
    # initialize the node
    rospy.init_node('RGB_detection')

    # subscribe to the input topics
    sub_rgb = message_filters.Subscriber(rgb_image_topic, Image)
    sub_depth = message_filters.Subscriber(depth_image_topic, Image)
    sub_camera_info = rospy.Subscriber(camera_info_topic, CameraInfo, camera_info_callback)
    combined_sub = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, 0.1)

    combined_sub.registerCallback(sync_callback)

    # create a publisher for the output topic for each object class
    for objectClass_id in classesDict:
        objectClass = classesDict[objectClass_id]
        output_topic = output_topic_prefix + objectClass
        pub = rospy.Publisher(output_topic, PoseStamped, queue_size=10)
        pub_dict[objectClass_id] = pub
    # run the node
    rospy.spin()

if __name__ == '__main__':
    listener()


######TODO
# handle multiple objects classes with global list and multiple publishers
# maybe transform frame from rgb to depth camera frame
# maybe use depth image rectified

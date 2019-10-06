#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
from arlorobot_msgs.msg import detection
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

u = 0
v = 0
label = ""

def detection_callback(msg):

    global label
    global u
    global v

    label = msg.label

    bb_x1 = msg.bb_x1
    bb_y1 = msg.bb_y1
    bb_x2 = msg.bb_x2
    bb_y2 = msg.bb_y2

    u = (((bb_x2 - bb_x1)/2) + bb_x1)
    v = (((bb_y2 - bb_y1)/2) + bb_y1)

def image_depth_callback(msg):

    # Convert ROS image to OpenCV image
    image_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    rospy.Subscriber("detection", detection, detection_callback)

    center_coordinates = (u,v)

    if label == "clock":

       depth_person = image_depth[u,v]

       if depth_person != 0:

           print (depth_person)

    cv2.waitKey(10)

def main():

    rospy.init_node('object_pose_callback')

    # Define your image topic
    image_depth = "/camera/depth/image_raw"

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_depth, Image, image_depth_callback)

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

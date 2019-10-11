#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# Detection message
from arlorobot_msgs.msg import detection
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2
# Import time to do delays
import time

import numpy as np

bridge = CvBridge()

i = 0
j = 0
label = ""

def detection_callback(msg):

    global label
    global i
    global j

    label = msg.label

    bb_x1 = msg.bb_x1
    bb_y1 = msg.bb_y1
    bb_x2 = msg.bb_x2
    bb_y2 = msg.bb_y2

    i = (((bb_x2 - bb_x1)/2) + bb_x1)
    j = (((bb_y2 - bb_y1)/2) + bb_y1)

def image_depth_callback(msg):

    # Convert ROS image to OpenCV image
    image_rect_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    rospy.Subscriber("detection", detection, detection_callback)

    # Center_object_pixel with pixel_ij

    center_object_pixel = (i,j)

    # Calculate z(depth) with pixel_ij

    z_pixel_ij = image_rect_depth[i,j]

    # Creating a vector V with dimensions of the pixel_ij

    v_pixel_ij = [i,j,1]

    # Calculate vector V transpose

    v_pixel_ij_T = np.transpose(v_pixel_ij)

    # Camera matrixK

    matrixK = [[538.400002091351, 0, 325.2479705810862], [0, 540.8584752417346, 252.4522833308824], [0, 0, 1]] # Matriu rectificada 3x3 La ultima columna fora

    # Calculate matrix K inverse

    matrixK_inv = np.linalg.inv(matrixK)

    # Calculate vector direction of the object

    direction_pixel_ij = np.dot(matrixK_inv,v_pixel_ij_T)

    print(label, center_object_pixel, z_pixel_ij, direction_pixel_ij)

    time.sleep(0.1)

def main():

    rospy.init_node('object_pose_rect_callback')

    # Define your image topic
    image_rect_depth = "/camera/depth/image_rect_raw"

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_rect_depth, Image, image_depth_callback)

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

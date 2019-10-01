#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from arlorobot_msgs.msg import detection
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2

def detection_callback(msg):

    label = msg.label

    bb_x1 = msg.bb_x1
    bb_y1 = msg.bb_y1
    bb_x2 = msg.bb_x2
    bb_y2 = msg.bb_y2

def main():

     	rospy.init_node('object_pose_callback')
        sub = rospy.Subscriber("detection", detection, detection_callback)

    	# Spin until ctrl + c
    	rospy.spin()

if __name__ == '__main__':
    main()

#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

def image_color_callback(msg):

    print("Received an image!")

    cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # Convert ROS image to OpenCV image

    cv2.namedWindow('rgb_img') # OpenCV window

    cv2.imshow('rgb_img', cv2_img) # Show the ros image_color inside the OpenCV window

    cv2.waitKey(1) # Delay

def main():

    rospy.init_node('ros_image_color_callback')

    # Define your image topic
    image_color = "/camera/rgb/image_color"

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_color, Image, image_color_callback)

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

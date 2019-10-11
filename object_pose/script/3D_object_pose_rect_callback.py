#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# message
from arlorobot_msgs.msg import DetectionArray
#from arlorobot_msgs.msg import Landmark
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2
# Import time to do delays
import time

import numpy as np

from queue import Queue

bridge = CvBridge()

class ObjectPose:

    def __init__(self):
        '''Initialize ros publisher and ros subscriber'''
        # topic where we publish
        #self.LandMarkPub = rospy.Publisher("Landmark", Landmark, queue_size = 1)
        # Define your image topic
        ImageRectDepth = "/camera/depth/image_rect_raw"
        # Set up your subscribers and define its callbacks
        self.ImageRectDepthSub = rospy.Subscriber(ImageRectDepth, Image, self.ImageRectDepth_Callback)
        #self.DetectionArraySub = rospy.Subscriber("DetectionArray", DetectionArray, self.DetectionArray_Callback)

    def ImageRectDepth_Callback(self, ImageRectDepth):

        ImageRectDepthArray = list()

        ImageRectDepthArray.append(ImageRectDepth)

        #if len(ImageRectDepthArray) == 10:
        #    ImageRectDepthArray.pop(0)

        print(len(ImageRectDepthArray))

    '''def DetectionArray_Callback(self, DetectionArray):

        ImageRectDepthStamp = ImageRectDepthArray.header.stamp.index(DetectionArray.header.stamp)

        # Convert ROS image to OpenCV image
        ImageRectDepthCv = bridge.imgmsg_to_cv2(ImageRectDepthStamp, desired_encoding="passthrough")

        label = DetectionArray.label
        bb_x1 = DetectionArray.bb_x1
        bb_y1 = DetectionArray.bb_y1
        bb_x2 = DetectionArray.bb_x2
        bb_y2 = DetectionArray.bb_y2

        i = (((bb_x2 - bb_x1)/2) + bb_x1)
        j = (((bb_y2 - bb_y1)/2) + bb_y1)

        print(i,j)'''


'''    # Center_object_pixel with pixel_ij

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

    time.sleep(0.1)'''

def main():

    '''Initializes ObjectPose'''
    ic = ObjectPose()
    rospy.init_node('ObjectPose')

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

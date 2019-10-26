#!/usr/bin/env python

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# message
from arlorobot_msgs.msg import DetectionArray
from arlorobot_msgs.msg import LandMark
from arlorobot_msgs.msg import LandMarkArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2
# Import time to do delays
import time

import numpy as np

bridge = CvBridge()

def Marker_function(self, figure, label, X_ij, Y_ij, Z_ij, ObjectArray):

    if figure == "CUBE":

        marker = Marker()
        marker.header = ObjectArray.header
        marker.type = marker.CUBE
        marker.id = ObjectArray.header.seq
        marker.scale.x = 100
        marker.scale.y = 100
        marker.scale.z = 100
        marker.color.a = 1.0

        if label == "chair":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif label == "diningtable":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif label == "sofa":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = X_ij
        marker.pose.position.y = Y_ij
        marker.pose.position.z = Z_ij
        marker.lifetime = rospy.Duration(3)

        return(marker)

    elif figure == "SPHERE":

        marker = Marker()
        marker.header = ObjectArray.header
        marker.type = marker.SPHERE
        marker.id = ObjectArray.header.seq
        marker.scale.x = 100
        marker.scale.y = 100
        marker.scale.z = 100
        marker.color.a = 1.0

        if label == "tvmonitor":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif label == "pottedplant":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif label == "bottle":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = X_ij
        marker.pose.position.y = Y_ij
        marker.pose.position.z = Z_ij
        marker.lifetime = rospy.Duration(3)

        return(marker)

    elif figure == "CYLINDER":

        marker = Marker()
        marker.header = ObjectArray.header
        marker.type = marker.CYLINDER
        marker.id = ObjectArray.header.seq
        marker.scale.x = 100
        marker.scale.y = 100
        marker.scale.z = 100
        marker.color.a = 1.0

        if label == "refrigerator":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif label == "oven":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif label == "microwave":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = X_ij
        marker.pose.position.y = Y_ij
        marker.pose.position.z = Z_ij
        marker.lifetime = rospy.Duration(3)

        return(marker)

class ObjectPose:

    def __init__(self):
        '''Initialize ros publisher and ros subscriber'''
        # topic where we publish
        self.LandMarkArrayPub = rospy.Publisher("LandMarkArray", LandMarkArray, queue_size = 1)
        self.MarkerArrayPub = rospy.Publisher("MarkerArray", MarkerArray, queue_size = 1)
        # Define your image topic
        ImageRectDepth = "/camera/depth/image_rect_raw"
        # Initialize ImageRectDepth list
        global ImageRectDepthArray
        ImageRectDepthArray = list()
        # Set up your subscribers and define its callbacks
        self.ImageRectDepthSub = rospy.Subscriber(ImageRectDepth, Image, self.ImageRectDepth_Callback)
        self.DetectionArraySub = rospy.Subscriber("DetectionArray", DetectionArray, self.DetectionArray_Callback)

    def ImageRectDepth_Callback(self, ImageRectDepth):

        ImageRectDepthArray.append(ImageRectDepth)

        if len(ImageRectDepthArray) == 100:
            ImageRectDepthArray.pop(0)

        # print(ImageRectDepthArray[-1].header.stamp)

    def DetectionArray_Callback(self, DetectionArray):

        for y in range(len(ImageRectDepthArray)):

#            print(ImageRectDepthArray[y].header.seq,DetectionArray.header.seq)

            if ImageRectDepthArray[y].header.seq == DetectionArray.header.seq:

                ImageRectDepthStamp = ImageRectDepthArray[y]

                # Convert ROS image to OpenCV image
                ImageRectDepthCv = bridge.imgmsg_to_cv2(ImageRectDepthStamp, desired_encoding="passthrough")

#                print(ImageRectDepthCv.shape)

                Det = DetectionArray.DetectionArray

                ObjectArray = LandMarkArray()
                Markers = MarkerArray()

                # copiar headers
                ObjectArray.header = ImageRectDepthArray[y].header

                for x in range(len(Det)):

                    label = Det[x].label
                    bb_x1 = Det[x].bb_x1
                    bb_y1 = Det[x].bb_y1
                    bb_x2 = Det[x].bb_x2
                    bb_y2 = Det[x].bb_y2

                    i = (((bb_x2 - bb_x1)/2) + bb_x1)
                    j = (((bb_y2 - bb_y1)/2) + bb_y1)

                    # Calculate z(depth) with pixel_ij
                    Z_ij = ImageRectDepthCv[j,i] # Veure perque diu que esta fora de rang (index)

                    # Creating a vector V transpose with dimensions of the pixel_ij
                    Vector_ij = [i,j,1]

                    # Creating a vector V transpose
                    VectorTranspose_ij = np.transpose(Vector_ij)

                    # Camera matrixK
                    MatrixK = [[556.3165893554688, 0, 324.6597875136467], [0, 558.7952880859375, 250.5356744470846], [0, 0, 1]] # Matriu rectificada 3x3, la ultima columna fora

                    # Calculate matrix K inverse
                    MatrixKInv = np.linalg.inv(MatrixK)

                    # Calculate vector direction of the object
                    VectorDirection_ij = np.dot(MatrixKInv,VectorTranspose_ij)

                    # Calculate x
                    X_ij = -(Z_ij*(VectorDirection_ij[0]/VectorDirection_ij[2])) # X inverted respect their frame

                    # Calculate y
                    Y_ij = (X_ij*(VectorDirection_ij[1]/VectorDirection_ij[0])) # Y inverted respect their frame

                    Object = LandMark()

                    Object.label = label
                    Object.X_ij = X_ij
                    Object.Y_ij = Y_ij
                    Object.Z_ij = Z_ij

                    if label == "chair" or label == "diningtable" or label == "sofa":

                        figure = "CUBE"

                        marker = Marker_function(self, figure, label, X_ij, Y_ij, Z_ij, ObjectArray)

                        Markers.markers.append(marker)

                    if label == "tvmonitor" or label == "pottedplant" or label == "bottle":

                        figure = "SPHERE"

                        marker = Marker_function(self, figure, label, X_ij, Y_ij, Z_ij, ObjectArray)

                        Markers.markers.append(marker)

                    if label == "refrigerator" or label == "oven" or label == "microwave":

                        figure = "CYLINDER"

                        marker = Marker_function(self, figure, label, X_ij, Y_ij, Z_ij, ObjectArray)

                        Markers.markers.append(marker)

                    ObjectArray.LandMarkArray.append(Object)

                self.LandMarkArrayPub.publish(ObjectArray)

                self.MarkerArrayPub.publish(Markers)

def main():

    '''Initializes ObjectPose'''
    rospy.init_node('ObjectPose')
    ic = ObjectPose()

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

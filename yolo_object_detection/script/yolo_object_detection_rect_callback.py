#!/usr/bin/env python

# importing os module
import os
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2

# yolo libraries and dependencies
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import argparse
import pickle as pkl

# message
from arlorobot_msgs.msg import Detection
from arlorobot_msgs.msg import DetectionArray

# Importing all yolo modules
import yolo_modules

# OpenCv CvBridge
bridge = CvBridge()

# Parse arguements to the detect module

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

# Global variables

# Absolute path
AbsolutePath = os.getenv("ARLO_PATH")

# cfgfile_path
CfgFilePath = os.path.join(AbsolutePath, "arlorobot/yolo_object_detection/script/cfg/" ,"yolov3.cfg")

# weightsfile_path
WeightsFilePath = os.path.join(AbsolutePath, "arlorobot/yolo_object_detection/script/weights/","yolov3.weights")

# classes_path
ClassesFilePath = os.path.join(AbsolutePath, "arlorobot/yolo_object_detection/script/classes/","coco.names")

# colors_path
ColorsFilePath = os.path.join(AbsolutePath, "arlorobot/yolo_object_detection/script/colors/","pallete_py2.pkl")


args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()

num_classes = 80
bbox_attrs = 5 + num_classes

model = yolo_modules.Darknet(CfgFilePath)
model.load_weights(WeightsFilePath)

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])

assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

classes = yolo_modules.load_classes(ClassesFilePath)
colors = pkl.load(open(ColorsFilePath, "rb"))

# Prepare image for inputting to the neural network

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

# Write object detected in a rectangle to the image

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    bb_x1 = c1[0].item()
    bb_y1 = c1[1].item()

    bb_x2 = c2[0].item()
    bb_y2 = c2[1].item()

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    box = [bb_x1, bb_y1, bb_x2, bb_y2]

    return img, label, box

class YoloNode:

    def __init__(self):
        '''Initialize ros publisher and ros subscriber'''
        # topic where we publish
        self.ObjectArrayPub = rospy.Publisher("DetectionArray", DetectionArray, queue_size = 1)
        # Define your image topic
        ImageRectColor = "/camera/rgb/image_rect_color"
    	# Set up your subscriber and define its callback
    	self.ImageRectColorSub = rospy.Subscriber(ImageRectColor, Image, self.ImageRectColor_Callback)

    def ImageRectColor_Callback(self, ImageRectColor):

        # Convert ROS image to OpenCV image
        ImageRectColorFrame = bridge.imgmsg_to_cv2(ImageRectColor, desired_encoding="passthrough")

        img, orig_im, dim = prep_image(ImageRectColorFrame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1,2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = model(Variable(img), CUDA)
        output = yolo_modules.write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

        output[:,[1,3]] *= ImageRectColorFrame.shape[1]
        output[:,[2,4]] *= ImageRectColorFrame.shape[0]

        ObjectArray = DetectionArray()

        # copiar headers
        ObjectArray.header = ImageRectColor.header

        for x in output:

           orig_im, label, box = write(x, orig_im)

           Object = Detection()

           Object.label = label
           Object.bb_x1 = box[0]
           Object.bb_y1 = box[1]
           Object.bb_x2 = box[2]
           Object.bb_y2 = box[3]

           ObjectArray.DetectionArray.append(Object)

        self.ObjectArrayPub.publish(ObjectArray)

def main():

    '''Initializes YoloNode'''
    rospy.init_node('YoloNode')
    ic = YoloNode()

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()

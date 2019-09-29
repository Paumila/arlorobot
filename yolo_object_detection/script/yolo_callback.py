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
from time import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random 
import argparse
import pickle as pkl

# Importing all yolo modules
import yolo_modules

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
basepath = os.getenv("ARLO_PATH")

# cfgfile_path
cfgfile_path = os.path.join(basepath, "arlorobot/yolo_object_detection/script/cfg/" ,"yolov3.cfg")

# weightsfile_path
weightsfile_path = os.path.join(basepath, "arlorobot/yolo_object_detection/script/weights/","yolov3.weights")

# classes_path
classesfile_path = os.path.join(basepath, "arlorobot/yolo_object_detection/script/classes/","coco.names")

# colors_path
colorsfile_path = os.path.join(basepath, "arlorobot/yolo_object_detection/script/colors/","pallete_py2.pkl")


args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()   
   
num_classes = 80
bbox_attrs = 5 + num_classes
   
model = yolo_modules.Darknet(cfgfile_path)
model.load_weights(weightsfile_path)
   
model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
    
assert inp_dim % 32 == 0 
assert inp_dim > 32

if CUDA:
    model.cuda()
            
model.eval()  

classes = yolo_modules.load_classes(classesfile_path)
colors = pkl.load(open(colorsfile_path, "rb"))

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
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    bb_x1 = c1[0].item()
    bb_y1 = c1[1].item()

    bb_x2 = c2[0].item()
    bb_y2 = c2[1].item()

    box = [bb_x1, bb_y1, bb_x2, bb_y2]

    object_coordinates = np.append(label, box)

    return img, object_coordinates


def image_color_callback(msg):

    # tiempo_ini = time()

    # Convert ROS image to OpenCV image

    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") 

    img, orig_im, dim = prep_image(frame, inp_dim)
            
    im_dim = torch.FloatTensor(dim).repeat(1,2) 
            
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    output = yolo_modules.write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
    output[:,[1,3]] *= frame.shape[1]
    output[:,[2,4]] *= frame.shape[0]
           
    for x in output:

       orig_im, object_coordinates = write(x, orig_im)

       # arlorobot_msgs_path
       arlorobot_msgs_path = os.path.join(basepath, "arlorobot/arlorobot_msgs/msg/" ,"detection.msg")

       detection = open(arlorobot_msgs_path,'w')
       
       for element in object_coordinates:

           print >> detection, element

       detection.close()
       
    cv2.imshow("frame", orig_im)

    cv2.waitKey(1)

""" Printing time to process the image

    tiempo_fin = time()
    tiempo_ejec = tiempo_fin - tiempo_ini
    print (tiempo_ejec) """

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

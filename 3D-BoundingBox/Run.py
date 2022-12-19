from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video",default=True,  action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    base_pts,box_3d=plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes
    return location,base_pts,box_3d

def run(frame):

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        # print('No previous model found, please train first!')
        exit()
    else:
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    curr_path = os.path.dirname(__file__)

    calib_file = curr_path+"/calib_cam_to_cam.txt"


    start_time = time.time()

    frame = cv2.resize(frame, (1242, 375))
    truth_img = frame
    img = np.copy(truth_img)
    yolo_img = np.copy(truth_img)

    detections = yolo.detect(yolo_img)
    base_ptslist = []
    ptslist = []
    locations = []


    for detection in detections:

        if not averages.recognized_class(detection.detected_class):
            continue
        
        detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)

        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = detection.box_2d
        detected_class = detection.detected_class

        input_tensor = torch.zeros([1,3,224,224]).cuda()
        input_tensor[0,:,:,:] = input_img

        [orient, conf, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        dim += averages.get_item(detected_class)

        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi

        location,base_pts,box_3d = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)
        
        base_ptslist.append(base_pts)
        ptslist.append(box_3d)
        locations.append(location)
        

    
    return base_ptslist,ptslist,locations


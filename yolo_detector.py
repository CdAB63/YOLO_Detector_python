#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:29:27 2022

@author: cdab63
"""

import os
import random
from ctypes import *
import numpy as np
import cv2
from structures import DETECTION, DETNUMPAIR, IMAGE, METADATA

class YOLODetector():
    
    def __init__(self,config_fn='detector.cfg',
                      data_fn='detector.data',
                      weights_fn='detector.weights'):
        # set the default class variables
        self.__configuration_file_name = config_fn
        self.__data_file_name          = data_fn
        self.__weights_file_name       = weights_fn
        
        # set the library
        if os.name == 'posix':
            cwd = os.path.dirname(__file__)
            self.__lib = CDLL(cwd + "/libdarknet.so", RTLD_GLOBAL)
        elif os.name == 'nt':
            cwd = os.path.dirname(__file__)
            os.environ['PATH'] = cwd + ';' + os.environ['PATH']
            os.add_dll_directory(cwd)
            # now see if there is a cuda toolkit
            cuda_location = 'c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA'
            if os.path.isdir(cuda_location):
                cuda_versions = [ sd for sd in os.listdir(cuda_location) if os.path.isdir(cuda_location+'/'+ sd) ]
                if cuda_versions:
                    cuda_versions.sort(reverse=True)
                    os.add_dll_directory(cuda_location+'/'+cuda_versions[0]+'/bin')
            self.__lib = CDLL("darknet.dll", RTLD_GLOBAL)
            
        # set library parameters
        self.__lib.network_width.argtypes = [c_void_p]
        self.__lib.network_width.restype = c_int
        self.__lib.network_height.argtypes = [c_void_p]
        self.__lib.network_height.restype = c_int
        
        self.__lib.network_width.argtypes = [c_void_p]
        self.__lib.network_width.restype = c_int
        self.__lib.network_height.argtypes = [c_void_p]
        self.__lib.network_height.restype = c_int

        self.copy_image_from_bytes = self.__lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

        self.predict = self.__lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.__lib.cuda_set_device
        self.init_cpu = self.__lib.init_cpu

        self.make_image = self.__lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.__lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.__lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.__lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_batch_detections = self.__lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self.free_ptrs = self.__lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.__lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.__lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.__lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = self.__lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.free_network_ptr = self.__lib.free_network_ptr
        self.free_network_ptr.argtypes = [c_void_p]
        self.free_network_ptr.restype = c_void_p

        self.do_nms_obj = self.__lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.__lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.__lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.__lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.__lib.get_metadata
        self.__lib.get_metadata.argtypes = [c_char_p]
        self.__lib.get_metadata.restype = METADATA

        self.load_image = self.__lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.__lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.__lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = self.__lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        self.network_predict_batch = self.__lib.network_predict_batch
        self.network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                                 c_float, c_float, POINTER(c_int), c_int, c_int]
        self.network_predict_batch.restype = POINTER(DETNUMPAIR)
        
    def network_width(self, net):
        return self.__lib.network_width(net)
    
    def network_height(self, net):
        return self.__lib.network_height(net)
    
    def bbox2points(self, bbox):
        """
        From bounding box yolo format
        to corner points cv2 rectangle
        """
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def class_colors(self, names):
        """
        Create a dict with one random BGR color for each
        class name
        """
        return {name: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)) for name in names}
    
    def load_network(self, config_file, data_file, weights, batch_size=1):
        """
        load model description and weights from config files
        args:
            config_file (str): path to .cfg model file
            data_file (str): path to .data model file
            weights (str): path to weights
        returns:
            network: trained model
            class_names
            class_colors
        """
        network = self.load_net_custom(
            config_file.encode("ascii"),
            weights.encode("ascii"), 0, batch_size)
        metadata = self.load_meta(data_file.encode("ascii"))
        class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]
        colors = self.class_colors(class_names)
        return network, class_names, colors
    
    def print_detections(self, detections, coordinates=False):
        print("\nObjects:")
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            if coordinates:
                print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
            else:
                print("{}: {}%".format(label, confidence))
                
    def draw_boxes(self, detections, image, colors):
        for label, confidence, bbox in detections:
            left, top, right, bottom = self.bbox2points(bbox)
            cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[label], 2)
        return image

    def decode_detection(self, detections):
        decoded = []
        for label, confidence, bbox in detections:
            confidence = str(round(confidence * 100, 2))
            decoded.append((str(label), confidence, bbox))
        return decoded
        
    # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Malisiewicz et al.
    def non_max_suppression_fast(self, detections, overlap_thresh):
        boxes = []
        for detection in detections:
            _, _, _, (x, y, w, h) = detection
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append(np.array([x1, y1, x2, y2]))
        boxes_array = np.array(boxes)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlap_thresh)[0])))
            # return only the bounding boxes that were picked using the
            # integer data type
        return [detections[i] for i in pick]
    
    def remove_negatives(self, detections, class_names, num):
        """
        Remove all classes with 0% confidence within the detection
        """
        predictions = []
        for j in range(num):
            for idx, name in enumerate(class_names):
                if detections[j].prob[idx] > 0:
                    bbox = detections[j].bbox
                    bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                    predictions.append((name, detections[j].prob[idx], (bbox)))
        return predictions

    def remove_negatives_faster(self, detections, class_names, num):
        """
        Faster version of remove_negatives (very useful when using yolo9000)
        """
        predictions = []
        for j in range(num):
            if detections[j].best_class_idx == -1:
                continue
            name = class_names[detections[j].best_class_idx]
            bbox = detections[j].bbox
            bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
            predictions.append((name, detections[j].prob[detections[j].best_class_idx], bbox))
        return predictions
    
    def detect_image(self, network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45):
        """
        Returns a list with highest confidence class and their bbox
        """
        pnum = pointer(c_int(0))
        self.predict_image(network, image)
        detections = self.get_network_boxes(network, image.w, image.h,
                                            thresh, hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        if nms:
            self.do_nms_sort(detections, num, len(class_names), nms)
        predictions = self.remove_negatives(detections, class_names, num)
        predictions = self.decode_detection(predictions)
        self.free_detections(detections, num)
        return sorted(predictions, key=lambda x: x[1])
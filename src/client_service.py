import socket
import numpy as np
import struct
import anyconfig
import munch
import sys
sys.path.append('../..')
from image_grabber import ImageGrabber
from uuid import getnode as get_mac
from lp_detector import LicensePlateDetector
from segmentation import SegmentCharacter

import netifaces
import pickle
import cv2
import time
import threading
from multiprocessing import Process, Queue

from utils import pad_or_truncate

class PreparingService(threading.Thread):
    def __init__(self, sending_queue, stop_dict, cfg, ext_ratio = 0.1):
        super(PreparingService, self).__init__()
        self.cfg = cfg
        self.device_id = get_mac()
        self.lp_detector = LicensePlateDetector(cfg.yolo_model)
        self.segmentation = SegmentCharacter()
        self.sending_queue = sending_queue
        self.image_grabber = ImageGrabber(cfg.camera.url, cfg.camera.usb)
        self.image_grabber.start()     
        self.ext_ratio = ext_ratio
        self.stop_dict = stop_dict

    def run(self):
        while not self.stop_dict["PreparingService"]:
            try:
                if not self.image_grabber.stop:
                    image = self.image_grabber.get_frame()
                    bboxes, labels, conf_scores = self.lp_detector.detect(image)
                    # Coordinate of characters in a license plate
                    char_coords = []
                    # Coordinate of license plate 
                    coord_boxes = []
                    for i, bbox in enumerate(bboxes):
                        conf_score = conf_scores[i]
                        coord_box = [int(val) for val in bbox]

                        width = coord_box[2] - coord_box[0]
                        height = coord_box[3] - coord_box[1]
                        coord_box[0] = max(0, int(coord_box[0] - width*self.ext_ratio))
                        coord_box[1] = max(0,int(coord_box[1] - height*self.ext_ratio))
                        coord_box[2] = min(image.shape[1], int(coord_box[2] + width*self.ext_ratio))
                        coord_box[3] = min(image.shape[0], int(coord_box[3] + height*self.ext_ratio))
                        
                        
                        lp_image = image[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2]]
                        # print(lp_image.shape)
                        processing_time = time.time()
                        char_coord_perbox = self.segmentation.segment(lp_image)
                        # print("Processing time:", time.time() - processing_time)

                        if len(char_coord_perbox) > 0 and len(char_coord_perbox) <= 10:
                            char_coord_perbox = pad_or_truncate(char_coord_perbox, 10)
                            coord_boxes.append(coord_box)
                            char_coords.append(char_coord_perbox)
                    
                    package = pickle.dumps({ "image": image, 
                                            "coord_boxes": coord_boxes, 
                                            "char_coords": char_coords,
                                            "deviceID": self.device_id })
                    self.sending_queue.put(package)
            except Exception as e:
                print(str(e))
        self.image_grabber.stop = True
        self.image_grabber.join()

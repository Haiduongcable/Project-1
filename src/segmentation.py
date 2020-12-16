import cv2 
import numpy as np 
import math
import time
import os 
import imutils
from utils import delete_bbox_overlap, bbox_in_boundary_image, sort_bbox
from utils import maximizeContrast, get_threshold, padding_rect, remove_bbox_noise
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class SegmentCharacter:
    def __init__(self):
        self.output_character_size = (12,28)
        self.width_fixed_size = 200

    def segment(self, lp_image):
        # Define type of license plate
        if lp_image.shape[0]/lp_image.shape[1] <= 0.6:
            type_lp = 'CarLong'
        else:
            type_lp = 'Square'
        
        #resize to format license plate
        lp_image = imutils.resize(lp_image, width = self.width_fixed_size)

        #threshold image
        gray_img = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
        gray_img = maximizeContrast(gray_img)
        threshed = get_threshold(gray_img)

        #find contours in image
        cnts, _ = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Find character in image
        if len(cnts) > 0:
            list_bbox = []
            for cnt in cnts:
                #Find bounding box 
                boundRect =(boxX, boxY, boxW, boxH) = cv2.boundingRect(cnt)
                topleft = (int(boundRect[0]), int(boundRect[1]))
                bottomright = (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3]))
                topright = (bottomright[0],topleft[1])
                bottomleft= (topleft[0],bottomright[1])
                fourpoint_rectangle = (topleft, topright, bottomright, bottomleft)

                #aspect Ratio: Ratio of width bounding box and height bounding box
                aspectRatio = boxW/float(boxH)
                #height Ratio: Ratio of height of bounding box and height of license plate
                heightRatio = boxH / float(lp_image.shape[0])
                keepAspectRatio = 0.05 < aspectRatio < 1.4
                keepHeight = 0.2 < heightRatio < 0.9
                n_pixel_white = np.sum(threshed[boxY:boxY+boxH, boxX:boxX+boxW] == 255)
                #Ratio of number pixels white in threshold roi and number of pixel in threshold roi
                whiteRatio = n_pixel_white / float(boxH * boxW)
                areaPolygon = boxW * boxH
               
                #Check if bouding box is of character
                if keepHeight and whiteRatio <= 0.95 and\
                        0.001 <= areaPolygon / (float(lp_image.shape[0]) * float(lp_image.shape[1])) <= 0.25\
                        and keepAspectRatio  and bbox_in_boundary_image(fourpoint_rectangle, lp_image):
                    boxCharacter = np.int0(fourpoint_rectangle)
                    boxCharacter = boxCharacter.tolist()
                    list_bbox.append(boxCharacter)
            
            #Delete bbox overlap with other
            print("Shape of bbox", np.shape(list_bbox))
            list_bbox_character = list_bbox.copy()
            for indexA in range(len(list_bbox) -1 ) :
                for indexB in range(indexA + 1, len(list_bbox)):
                    delete_bbox_overlap(indexA, indexB, list_bbox, list_bbox_character)
        # Remove bounding box is too small or have special shape
        list_bbox_character = remove_bbox_noise(list_bbox_character)
        list_extendbbox = []
        #Padding bounding box to have more large image of character
        for bbox in list_bbox_character:
            list_extendbbox.append(padding_rect(bbox))
        #Sorted bbox right to left, top to down
        if len(list_extendbbox) >= 3 and len(list_extendbbox) <= 10:
            list_sorted = sort_bbox(type_lp, list_extendbbox)
            # print("Checking: ",len(list_sorted))
            return list_sorted
        elif len(list_extendbbox) < 3:
            return list_extendbbox
        else:
            return []
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
from utils import get_contour_precedence, square, sort


class SegmentCharacter:
    def __init__(self):
        self.output_character_size = (12,28)
        self.width_fixed_size = 600

    def segment(self, img):
        type_lp = 'CarLong'
        img = imutils.resize(img, width = self.width_fixed_size)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        scale1 = 0.005
        scale2 = 0.1
        area_condition1 = area * scale1
        area_condition2 = area * scale2
        # global thresholding
        ret1,th1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

        # Otsu's thresholding
        ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(imgray,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cropped = []
        list_bbox = []
        for cnt in contours:
            (x,y,w,h) = cv2.boundingRect(cnt)
            if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.3 and h/w > 0.3):
                list_bbox.append([(x,y),(x + w, y), (x + w, y + h), (x, y + h)])
                
        print("Shape of bbox raw:",np.shape(list_bbox))
        list_bbox_character = list_bbox.copy()
        for indexA in range(len(list_bbox) -1 ) :
            for indexB in range(indexA + 1, len(list_bbox)):
                delete_bbox_overlap(indexA, indexB, list_bbox, list_bbox_character)
                
        print("Shape bbox character", np.shape(list_bbox_character))
        list_bbox_character = remove_bbox_noise(list_bbox_character)
        print("Shape output bbox character", np.shape(list_bbox_character))
        
        list_extendbbox = []
        #Padding bounding box to have more large image of character
        for bbox in list_bbox_character:
            if (bbox_in_boundary_image(bbox,img)):
                list_extendbbox.append(padding_rect(bbox))
        imagedraw = img.copy()
        for bbox in list_extendbbox:
            cv2.rectangle(imagedraw, (bbox[0][0],bbox[0][1]), (bbox[2][0],bbox[2][1]), (255, 0, 0), 2)
        cv2.imwrite('ututut.png', imagedraw)
        #Sorted bbox right to left, top to down
        if len(list_extendbbox) >= 3 and len(list_extendbbox) <= 10:
            list_sorted = sort_bbox(type_lp, list_extendbbox)
            # print("Checking: ",len(list_sorted))
            return list_sorted
        elif len(list_extendbbox) < 3:
            return list_extendbbox
        else:
            return []
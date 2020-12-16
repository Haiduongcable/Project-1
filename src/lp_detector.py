import numpy as np 
import os 
from darknet import *
import cv2
import time
import glob
import imutils
import anyconfig
import munch
class LicensePlateDetector():
    def __init__(self, config):
        self.config = config
        self.conf_threshold = 0.5
        self.nms_threshold = 0.45
        self.hier_threshold = 0.5
        self.net = load_net(str.encode(config.config), str.encode(config.weight), 0)
        self.meta = load_meta(str.encode(config.meta))
        self.darknet_image_placeholder = make_image(network_width(self.net), network_height(self.net), 3)
        self.size = 416
        
    def resize_padding(self, im):
        im_h, im_w = im.shape[:2]
        longer_edge = max(im_h, im_w)
        ratio = longer_edge/self.size
        new_h = int(im_h/ratio)
        new_w = int(im_w/ratio)
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
		
        delta_w = self.size - new_w
        delta_h = self.size - new_h
        top, bottom = delta_h//2, delta_h - delta_h//2
        left, right = delta_w//2, delta_w - delta_w//2
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        return new_im, ratio, top, left

    def detect(self, image):
        image, ratio, top_pad, left_pad = self.resize_padding(image.copy())
        copy_image_from_bytes(self.darknet_image_placeholder, image.tobytes())
        res = detect_image(net=self.net,
                    meta=self.meta,
                    im=self.darknet_image_placeholder,
                    thresh=self.conf_threshold, 
                    hier_thresh=self.hier_threshold,
                    nms=self.nms_threshold)
        boxes = []
        labels = []
        confident_scores = []
        for r in res:
            label, confident_score, box = r
            x, y, w, h = box
            xmin = (x - w//2 - left_pad)*ratio
            ymin = (y - h//2 - top_pad)*ratio
            xmax = (x + w//2 - left_pad)*ratio
            ymax = (y + h//2 - top_pad)*ratio
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            confident_scores.append(confident_score)
        boxes = np.array(boxes).astype('int').reshape(-1, 4)
        return boxes, labels, confident_scores
    
    def draw_bboxes(self, image, boxes):
        '''
        image: opencv image shape h x w x d
        boxes: np.int shape 1 x 4: topleft, bottomright
        '''
        topleft = (boxes[0][0], boxes[0][1])
        bottomright = (boxes[0][2], boxes[0][3])
        drawed_image = cv2.rectangle(image, topleft, bottomright, (0,0,255), 2)
        return drawed_image
        

if __name__ == "__main__":
    cfg = anyconfig.load("config.yaml")
    cfg = munch.munchify(cfg)
    detector = LicensePlateDetector(cfg.yolo_model)
    # image = cv2.imread('/home/duongnh/Downloads/loat-xe-may-bien-so-dep-gia-sieu-dat-tuan-qua-1-1558138698605.jpg')
    # boxes, labels, confident_scores = detector.detect(image)
    # print("Output boxes shape", np.shape(boxes))
    # print("Label",labels)
    # print("Confidence score ", confident_scores)
    # imagedraw = detector.draw_bboxes(image, boxes)
    # cv2.imwrite("Ouput.jpg",imagedraw)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        boxes, labels, confident_scores = detector.detect(frame)
        print("Output boxes shape", np.shape(boxes))
        print("Label",labels)
        print("Confidence score ", confident_scores)
        if np.shape(boxes) == (1,4):
            imagedraw = detector.draw_bboxes(frame, boxes)
            cv2.imshow("Ouput",imagedraw)
        else:
            cv2.imshow("Ouput",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
    cv2.destroyAllWindows() 
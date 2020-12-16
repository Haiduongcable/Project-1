import numpy as np 
import cv2 
import os 
import anyconfig
import tensorflow as tf 
from recognition import ClassifyService
from yolo_opencv import Yolo_opencv
from segmentation import SegmentCharacter
import time
from utils import pad_or_truncate
import munch
from utils import crop_image

# def draw_char_bbox(image, char_coord_perbox):
#     for bbox in char_coord_perbox:
#         topleft = (bbox[0][0], bbox[0][1])
#         bottomright = (bbox[2][0], bbox[2][1])
#         print(topleft, bottomright, np.shape(char_coord_perbox))
#         cv2.rectangle(image, topleft, bottomright, (0,0,255), 1)
#     return image

#Test 
# image = cv2.imread('/home/duongnh/Downloads/loat-xe-may-bien-so-dep-gia-sieu-dat-tuan-qua-1-1558138698605.jpg')
# image_seg = image.copy()
yolo_model = Yolo_opencv()
segmentation = SegmentCharacter()

cfg = anyconfig.load("/home/duongnh/Documents/Project1/src/config_Cnn.yaml")
cfg = munch.munchify(cfg)
recognition = ClassifyService(cfg)



def test(image):
    image_seg = image.copy()
    output_yolo, bboxes =  yolo_model.detect(image)
    print(len(bboxes))
    cv2.imwrite("output_yolo.jpg", output_yolo)
    char_coords = []
    # Coordinate of license plate 
    coord_boxes = []
    ext_ratio = 0.1
    for i, bbox in enumerate(bboxes):
        coord_box = []
        coord_box.append(bbox[0])
        coord_box.append(bbox[1])
        width = bbox[2]
        height = bbox[3]
        coord_box.append(bbox[0] + width)
        coord_box.append(bbox[1] + height)
        coord_box[0] = max(0, int(coord_box[0] - width*ext_ratio))
        coord_box[1] = max(0,int(coord_box[1] - height*ext_ratio))
        coord_box[2] = min(image_seg.shape[1], int(coord_box[2] + width*ext_ratio))
        coord_box[3] = min(image_seg.shape[0], int(coord_box[3] + height*ext_ratio))
        
        lp_image = image_seg[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2]]
        cv2.imwrite("Lp_image.jpg", lp_image)
        time_start = time.time()
        char_coord_perbox = segmentation.segment(lp_image)
        print("Processing time: ", time.time() - time_start)
        if len(char_coord_perbox) > 0 and len(char_coord_perbox) <= 10:
            char_coord_perbox = pad_or_truncate(char_coord_perbox, 10)
            coord_boxes.append(coord_box)
            char_coords.append(char_coord_perbox)

    recognition_image, bboxes_nms = recognition.predict(image_seg, coord_boxes, char_coords)
    cv2.imwrite("Final_output.jpg",recognition_image)

image = cv2.imread("/home/duongnh/Documents/biensoxe/bien4.jpg")
test(image)
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     test(frame)
#     #boxes, labels, confident_scores = detector.detect(frame)
#     # print("Output boxes shape", np.shape(boxes))
#     # print("Label",labels)
#     # print("Confidence score ", confident_scores)
#     # if np.shape(boxes) == (1,4):
#     #     imagedraw = detector.draw_bboxes(frame, boxes)
#     #     cv2.imshow("Ouput",imagedraw)
#     # else:
#     #     cv2.imshow("Ouput",frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#             break
# cv2.destroyAllWindows() 
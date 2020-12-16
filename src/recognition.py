import numpy as np

import cv2
import imutils
import tensorflow as tf
import threading
import anyconfig
import munch
import time
from model_Cnn import CharacterRecognition
from utils import four_point_transform, resize_and_pad


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 
13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'K', 19: 'L',
20: 'M', 21: 'N', 22: 'P', 23: 'R', 24: 'S', 25: 'T', 26: 'U', 
27: 'V', 28: 'X', 29: 'Y', 30: 'Z'}

def crop_image(image,list_coordinate):
    """
        Crop character in lp image
    """
    list_character = []
    lp_image = imutils.resize(image,width = 200)
    for bbox in list_coordinate:
        if bbox[0][0] == bbox[0][1] == bbox[1][0] == bbox[1][1]:
            break

        pts = np.array([(bbox[0][0],bbox[0][1]),
                (bbox[1][0],bbox[1][1]),
                (bbox[2][0],bbox[2][1]),
                (bbox[3][0],bbox[3][1])],dtype = "float32")
        
        warped = four_point_transform(lp_image,pts)

        # _,warped = cv2.threshold(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # warped = cv2.resize(warped,(12,28))
        warped =  resize_and_pad(warped, (28,28), padColor= 255)
        warped = warped / 255.0

        # warped = warped[..., None]
        list_character.append(warped)
    return list_character


class ClassifyService():
    def __init__(self, cfg):
        self.classify_model = CharacterRecognition()
        input_shape = (None, 28, 28, 3)
        self.classify_model.build(input_shape)
        # print(self.classify_model.summary())
        self.classify_model.load_weights(cfg.checkpoint)
        self.classify_model(np.ones((10, 28, 28, 3)).astype("float32"))

    def predict(self, image, coords_boxes, char_coords):
        list_class = []
        # #Coord of lp images
        for i  in  range(len(char_coords)):
            #list chars coord in a lp_image
            char_coord = char_coords[i]

            coord_box = coords_boxes[i]
            lp_image = image[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2]]
            #Coord in lp image and get lp imgage
            char_images = crop_image(lp_image, char_coord)
            char_images = np.array(char_images).astype("float32")
            
            len_chars = len(char_images)

            add_tensor = np.ones((10 - len_chars, 28, 28, 3)).astype("float32")

            add_char_images = np.concatenate((char_images, add_tensor))

            outputs = self.classify_model(add_char_images, training = False)
            results = tf.math.argmax(outputs, axis= -1)
            results = list(results.numpy())[0:len_chars]
            #Get name character
            class_names = [class_dict[t] for t in results]

            #Get name lp
            lp_name = "".join(class_names)
            list_class.append(lp_name)

            lp_image_resize_400 = imutils.resize(lp_image, width = 200)

            list_character_coord = np.asarray(char_coord)

            for bbox in list_character_coord:
                cv2.drawContours(lp_image_resize_400,[bbox],0,(0,0,255),2)

            image[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2]] = cv2.resize(lp_image_resize_400,(lp_image.shape[1],lp_image.shape[0]))
            y = coord_box[1] - 15 if coord_box[1] - 15 > 15 else coord_box[1] + 15
            cv2.putText(image, lp_name, (coord_box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #draw lp
            cv2.rectangle(image, (coord_box[0], coord_box[1]), (coord_box[2], coord_box[3]), (0, 0, 255), 1)
        return image, list_class
import os
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import imutils
import time
#from sklearn.utils import shuffle

def resize_and_pad(img, size, padColor= 0):
    """
        Padding image and resize
    """
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def load_data(dataset_path, training=True):
    """
        Load classify dataset

        data_path: Path to dataset
    """
    output = []
    class_names = os.listdir(dataset_path)
    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    images = []
    labels = []
    for folder in class_names:
        label = class_names_label[folder]
        for file in tqdm(os.listdir(os.path.join(dataset_path, folder))):
            img_path = os.path.join(os.path.join(dataset_path, folder), file)
            image = cv2.imread(img_path, 0)

            if not training:
                _, image = cv2.threshold(image ,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                image = resize_and_pad(image, (28, 28))

            image = image /255.0

            images.append(image)
            labels.append(label)
    
    images = np.array(images, dtype= "float32")
    images = images[..., None]
    labels = np.array(labels, dtype= "uint8")
    labels = tf.one_hot(labels, len(class_names))
    return (images, labels)


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return cv2.resize(warped,(12,28))

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

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


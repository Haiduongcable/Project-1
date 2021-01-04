import cv2 
import numpy as np 
import math
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
from heapq import nsmallest
from skimage.filters import threshold_local
from skimage import measure
import imutils
from sklearn.cluster import KMeans




# Time: 16/12/20 09:28 Result: 64A04075
def querry_t(string_lp, string_time):
    dir_querry = os.listdir('/home/duongnh/Documents/Project1/Database/save_lpname')
    datetime_str = ""
    for filename in dir_querry:
        file = open('/home/duongnh/Documents/Project1/Database/save_lpname/' + filename)
        for line in file:
            if (len(line) > 10):
                if line.find(string_lp) != -1:
                    end_index = line.find(string_lp) - 8
                    datetime_str = line[6:end_index]
                    break
    return datetime_str

    

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


def remove_bbox_noise(ls_bbox):
    '''
    Remove the bboxs that is too small or have special shape.
    
    '''
    try:
        ls_height = []
        for bbox in ls_bbox:
            ls_height.append(bbox[3][1] - bbox[0][1])
        index_bbox = []
        mean_height = sum(ls_height)/len(ls_height)
        for index in range(len(ls_height)):
            if mean_height*1.5 >= ls_height[index] or ls_height[index] >= mean_height*0.65:
                index_bbox.append(index)
        output_bbox = []
        for i in index_bbox:
            output_bbox.append(ls_bbox[i])
        return output_bbox
    except:
        return ls_bbox


#threshold image
def get_threshold(gray_img, inv = True):
    gray_img = cv2.GaussianBlur(gray_img,(5,5),0)
    if inv:
       _, threshed = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
       _, threshed = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    return threshed

#maximum Contrast 
def maximizeContrast(imgGrayscale):
    '''
    Increase constrast in image
    '''
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

#delete bbox is overlap with others
def delete_bbox_overlap(indexA, indexB,list_bbox,list_bbox_Copy):
    '''
    indexA: index of bbox A
    indexB: index of bbox B
    if bbox A and bbox B overlap, detele the smallest
    '''
    bboxA = list_bbox[indexA]
    bboxB = list_bbox[indexB]
    polygon_A = Polygon(bboxA)
    polygon_B = Polygon(bboxB)
    intersec_area = polygon_A.intersection(polygon_B).area
    area_polygonA = polygon_A.area
    area_polygonB = polygon_B.area
    ratio_overlap = intersec_area / min(area_polygonA,area_polygonB)
    if ratio_overlap >= 0.3 and area_polygonA < area_polygonB:
        if bboxA in list_bbox_Copy:
            list_bbox_Copy.remove(bboxA)
    elif ratio_overlap >= 0.3 and area_polygonA > area_polygonB:
        if bboxB in list_bbox_Copy:
            list_bbox_Copy.remove(bboxB)


def second_smallest(numbers,index):
    return nsmallest(index, numbers)[-1]

def average(lst): 
    return sum(lst) / len(lst) 

#get bbox only in boundary of license
def bbox_in_boundary_image(box,image):
    '''
    Checking if bbox has coordinates outside image
    '''
    count_zero = 0
    count_right = 0
    count_bottom = 0
    count_outside = 0
    for point in box:
        if 0 in point:
            count_zero += 1
        if point[1] >= image.shape[0] -2:
            count_outside += 1
        if point[0] >= image.shape[1] -2:
            count_right += 1
        if point[0] <0 or point[1] <0:
            count_outside += 1
            
    if count_zero >=2 or count_bottom >= 1 or count_right >= 1 or count_outside >=1:
        return False
    else:
        return True


def kmean_2_cluster(list_y_center):
    X = np.reshape(list_y_center,(len(list_y_center),1))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    return kmeans.labels_.astype(int).tolist()

def sort_X_right2left(list_bbox):
    '''
    sort bbox from right to left 
    '''
    list_sortbbox = []
    list_xcenter = []
    list_ycenter = []
    for bbox in list_bbox:
        list_xcenter.append(int((bbox[0][0] + bbox[1][0] +\
                        bbox[2][0] + bbox[3][0])/4))
        list_ycenter.append(int((bbox[0][1] +  bbox[1][1] +\
                        bbox[2][1]  + bbox[3][1])/4))
    sorted_xcenter = list_xcenter.copy()
    sorted_xcenter.sort()
    for xcenter in sorted_xcenter:
            list_sortbbox.append(list_bbox[list_xcenter.index(xcenter)])
    return list_sortbbox

#sort character in license plate left to right, up to down
def sort_bbox(type_lp,list_bbox):
    '''
    There are two types of license plate: CarLong and Square
    Sort bbox from top to bottom and right to left
    '''
    try:
        list_xcenter = []
        list_ycenter = []
        for bbox in list_bbox:
            list_xcenter.append(int((bbox[0][0] + bbox[1][0] +\
                            bbox[2][0] + bbox[3][0])/4))
            list_ycenter.append(int((bbox[0][1] +  bbox[1][1] +\
                            bbox[2][1]  + bbox[3][1])/4))
        if type_lp == 'CarLong':
            list_sort = sort_X_right2left(list_bbox)
            return list_sort
        else:
            # Use Kmean cluster bounding box to 2 list (Top and bottom) with type: Square
            list_index_Y = kmean_2_cluster(list_ycenter)
            list_index_1 = [i for i in range(len(list_ycenter)) if list_index_Y[i] == 1]
            list_index_2 = [i for i in range(len(list_ycenter)) if list_index_Y[i] == 0]
            if average([list_ycenter[i] for i in list_index_1]) <=\
                    average([list_ycenter[i] for i in list_index_2]):
                list_up_BBox = sort_X_right2left([list_bbox[i] for i in list_index_1])
                list_down_BBox = sort_X_right2left([list_bbox[i] for i in list_index_2]) 
                return list_up_BBox + list_down_BBox
            else:
                list_up_BBox = sort_X_right2left([list_bbox[i] for i in list_index_2])
                list_down_BBox = sort_X_right2left([list_bbox[i] for i in list_index_1])
                return list_up_BBox + list_down_BBox
    except:
        return []

def pad_or_truncate(some_list, target_len):
    return some_list + [[[0,0], [0, 0], [0, 0], [0, 0]]]*(target_len - len(some_list))

def convert_format_list(array_a):
    array_a = np.int0(array_a)
    list_a = array_a.tolist()
    return list_a

def padding_rect(sort_point, config_ratio = 0.1):
    '''
    Sort_point: List bbox after sorting from top to bottom, right to left
    Extend size of bbox to get more large images of character
    '''
    sort_point = np.array(sort_point)
    vecto1 = sort_point[0] - sort_point[2]
    vecto_padding1 = vecto1 * (1 + config_ratio)
    vecto2 = sort_point[1] - sort_point[3]
    vecto_padding2 = vecto2 * (1 + config_ratio)
    padding_topleft = sort_point[2] + vecto_padding1
    padding_bottomright = sort_point[0] - vecto_padding1
    padding_topright = sort_point[3] + vecto_padding2
    padding_bottomleft = sort_point[1] - vecto_padding2
  
    paddingpoint = [convert_format_list(padding_topleft),convert_format_list(padding_topright), 
                    convert_format_list(padding_bottomright), convert_format_list(padding_bottomleft)]
    return paddingpoint

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


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def square(img):
    """
    This function resize non square image to square one (height == width)
    :param img: input image as numpy array
    :return: numpy array
    """

    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image


def sort(vector):
    sort = True
    while (sort == True):

        sort = False
        for i in range(len(vector) - 1):
            x_1 = vector[i][0]
            y_1 = vector[i][1]

            for j in range(i + 1, len(vector)):

                x_2 = vector[j][0]
                y_2 = vector[j][1]

                if (x_1 >= x_2 and y_2 >= y_1):
                    tmp = vector[i]
                    vector[i] = vector[j]
                    vector[j] = tmp
                    sort = True

                elif (x_1 < x_2 and y_2 > y_1):
                    tmp = vector[i]
                    vector[i] = vector[j]
                    vector[j] = tmp
                    sort = True
    return vector
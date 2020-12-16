
# Loading the required python modules 
import pytesseract # this is tesseract module 
import matplotlib.pyplot as plt 
import cv2 # this is opencv module 
import glob 
import os

class TesseractRecognition:
    def __init__(self):
        #self.config = ='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        # filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "") 
        # predicted_license_plates.append(filter_predicted_result) 
        self.custom_config = r'--oem 3 --psm 6'
        #'-c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3'
        
    def recogn(self, image):
        predicted_result = pytesseract.image_to_string(image, lang ='eng', config = self.custom_config)
        filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
        string_db = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        new_str = ""
        for index, char_i in enumerate(filter_predicted_result):
            #print(char_i)
            if char_i in string_db:
                #print(type(filter_predicted_result[index]))
                new_str = new_str + filter_predicted_result[index]
        return new_str
                
    
if __name__ == "__main__":
    tesseract_recog = TesseractRecognition()
    image = cv2.imread('/home/duongnh/Documents/biensoxe/anhcut2.jpg')
    #result = tesseract_recog.recogn(image)
    #print(result)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = hsv[:,:,2] - 50
    image_dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("ImageDark.jpg", image_dark )
    
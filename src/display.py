import tkinter as tk
import threading, queue
from tkinter import *
from camera import Camera 
import cv2
import PIL
import PIL.ImageTk
from PIL import ImageTk
from PIL import Image
import time
import cv2
import time
import glob
import imutils
import anyconfig
import munch
import numpy as np
import tkinter.font as font
import os 
from segmentation import SegmentCharacter
from yolo_opencv import Yolo_opencv
import time
from utils import pad_or_truncate
import munch
from utils import crop_image
from recognition import ClassifyService
import requests
from datetime import datetime
from tesseract_recognition import TesseractRecognition
from utils import querry_t
from threading import Thread
from mysql_database import MySQL_Database

class Display:
    def __init__(self):
        self.webcam = self.videocapture()
        self.ip_url = "http://192.168.43.67:8080/shot.jpg"
        self.height_window = 680
        self.width_window = 1540
        self.window = self.create_window()
        self.canvas = self.create_canvas()
        self.button = self.create_button()
        self.photo = None
        self.count = 0
        self.frame_save = None
        self.yolomodel = Yolo_opencv()
        self.segmentation_image = None
        self.segmentation = SegmentCharacter()
        cfg = anyconfig.load("config_Cnn.yaml")
        cfg = munch.munchify(cfg)
        self.recognition = ClassifyService(cfg)
        self.bboxes_yolo = None
        self.image_result_yolo = None
        self.lp_recognition = None
        self.tesseract = TesseractRecognition()
        #Option
        self.type_camera = "IP" #default "IP"
        self.modelRecognition = "Segmentation Recognition" #default "Segmentation Recognition"
        #Database
        self.database = MySQL_Database()
        
        # Label Tkinter 
        self.label_lp = None

        
    def tesseract_recognition(self, image, bboxes):
        image_recog = image.copy()
        license_plate = cv2.resize(image, None, fx = 2, fy = 2,
                                        interpolation = cv2.INTER_CUBIC)
        gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        blur_license_plate = cv2.GaussianBlur(gray_license_plate, (5, 5), 0)
        ext_ratio = 0.1
        coord_boxes = []
        lp_result_l = []
        for i, bbox in enumerate(bboxes):
            coord_box = []
            coord_box.append(bbox[0])
            coord_box.append(bbox[1])
            width = bbox[2]
            height = bbox[3]
            coord_box.append(bbox[0] + width)
            coord_box.append(bbox[1] + height)
            coord_box[0] = max(0, int(coord_box[0] + width*ext_ratio))
            coord_box[1] = max(0,int(coord_box[1] + height*ext_ratio))
            coord_box[2] = min(image.shape[1], int(coord_box[2] - width*ext_ratio))
            coord_box[3] = min(image.shape[0], int(coord_box[3] - height*ext_ratio))
            lp_image = blur_license_plate[coord_box[1]:coord_box[3], coord_box[0]:coord_box[2]]
            time_start = time.time()
            result_lp = self.tesseract.recogn(lp_image)
            lp_result_l.append(result_lp)
        return image, lp_result_l
                
    def segmentation_recognition(self, image, bboxes):
        image_seg = image.copy()
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
            char_coord_perbox = self.segmentation.segment(lp_image)
            print("Processing time: ", time.time() - time_start)
            if len(char_coord_perbox) > 0 and len(char_coord_perbox) <= 10:
                char_coord_perbox = pad_or_truncate(char_coord_perbox, 10)
                coord_boxes.append(coord_box)
                char_coords.append(char_coord_perbox)
                
        result_recognition_image, listclass_licenseplate = self.recognition.predict(image_seg, coord_boxes, char_coords)
        
        return result_recognition_image , listclass_licenseplate
        
        
    def update_frame(self):
        if self.type_camera == "Webcam":
            ret, frame = self.webcam.read()
        else:
            r = requests.get(self.ip_url)
            img_arr = np.array(bytearray(r.content),dtype=np.uint8)
            frame = cv2.imdecode(img_arr,-1)
            frame = cv2.resize(frame, dsize = None, fx = 0.7, fy = 0.7)
            
        self.frame_save = frame
        input_yolo = frame.copy()
        self.image_result_yolo, self.bboxes_yolo = self.yolomodel.detect(input_yolo)
        
        image_imshow = cv2.cvtColor(self.image_result_yolo, cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image_imshow))
        self.canvas.create_image(0,0, image = self.photo, anchor=tk.NW)
        self.window.after(30,self.update_frame)
        self.count +=1

    def videocapture(self):
        video = cv2.VideoCapture(0)
        return video
    
    def create_window(self):
        window = tk.Tk()
        window.title("Vietnamese License Plate Recognition System")
        window.geometry("1480x680")
        return window
    
    def create_canvas(self):
        canvas_w = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
        canvas_h = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        canvas = Canvas(self.window, width = canvas_w, height= canvas_h , bg= "red")
        canvas.place(relx=0.2, rely=0.4, anchor="c")
        return canvas
    
        
    def get_lp_recognition(self):
        if self.modelRecognition == "Tesseract":
            result_recognition_image , self.lp_recognition  = self.tesseract_recognition(self.frame_save, self.bboxes_yolo)
            #print(self.lp_recognition)
        else:
            result_recognition_image , self.lp_recognition = self.segmentation_recognition(self.frame_save, self.bboxes_yolo)
            
    def Result_VehicleIN(self):
        print("Tap button")
        # Imshow saved
        myFont = font.Font(family="Helvetica",size=36,weight="bold")
        var = StringVar()
        label_saved = Label(textvariable=var , width = 10, height = 3, font =  myFont)
        var.set("Saved")
        label_saved.place(relx = 0.8, rely = 0.03, anchor = "c")
        '''
        Imshow label with saved image
        '''
        image_imshow = cv2.resize(self.frame_save,dsize=None, fx = 0.5, fy = 0.5)
        image_imshow = cv2.cvtColor(image_imshow, cv2.COLOR_BGR2RGB)
        image_imshow = Image.fromarray(image_imshow)
        image_imshow = ImageTk.PhotoImage(image_imshow)
        label_image = Label(image = image_imshow)
        label_image.image = image_imshow
        label_image.place(relx = 0.8, rely = 0.3, anchor = "c")
        
        #get date time
        now = datetime.now()
        datetime_str = now.strftime("%d/%m/%y %H:%M")
        
        # Imwrite
        cv2.imwrite("/home/duongnh/Documents/Project1/Database/save_frame/"\
                        + str(self.count) + '_image.jpg', self.frame_save)
        
        if (len(self.bboxes_yolo) != 0):
            cv2.imwrite("/home/duongnh/Documents/Project1/Database/save_result/"\
                                 + str(self.count) + '_image.jpg',self.image_result_yolo)
            
            file = open("/home/duongnh/Documents/Project1/Database/save_lpname/" + str(self.count) + "_image.txt", "w")
            
            string_imshow = ""
            
            self.get_lp_recognition()
            
            for lp_name in self.lp_recognition:
                print(type(datetime_str), type(lp_name))
                if datetime_str != "" and lp_name != None:
                    file.write("Time: " + datetime_str + " Result: " + lp_name +  "\n")
                    string_imshow += lp_name + " " + "\n"
                # datetime_in = querry_t(lp_name, datetime_str)
                file.close()
                break
            
            #Insert database
            self.database.insert_Plate_IN(self.lp_recognition[0], datetime_str)
            string_imshow += "\n Time in: " + datetime_str
            
            #lable imshow lpname
            myFont = font.Font(family="Helvetica",size=18,weight="bold")
            var_lp = StringVar()
            self.label_lp = Label(self.window, textvariable=var_lp , width = 40, height = 12, font =  myFont)
            var_lp.set(string_imshow)
            self.label_lp.place(relx = 0.8, rely = 0.8, anchor = "c")
 
        else:
            myFont = font.Font(family="Helvetica",size=18,weight="bold")
            var_error = StringVar()
            self.label_lp = Label(self.window, textvariable=var_error , width = 40, height = 12, font =  myFont)
            var_error.set("No license plate in image")
            self.label_lp.place(relx = 0.8, rely = 0.8, anchor = "c")
            
            
    def Result_VehicleOUT(self):
        print("Tap button")
        # Imshow saved
        myFont = font.Font(family="Helvetica",size=36,weight="bold")
        var = StringVar()
        label_saved = Label(textvariable=var , width = 10, height = 3, font =  myFont)
        var.set("Saved")
        label_saved.place(relx = 0.8, rely = 0.03, anchor = "c")
        '''
        Imshow label with saved image
        '''
        image_imshow = cv2.resize(self.frame_save,dsize=None, fx = 0.5, fy = 0.5)
        image_imshow = cv2.cvtColor(image_imshow, cv2.COLOR_BGR2RGB)
        image_imshow = Image.fromarray(image_imshow)
        image_imshow = ImageTk.PhotoImage(image_imshow)
        label_image = Label(image = image_imshow)
        label_image.image = image_imshow
        label_image.place(relx = 0.8, rely = 0.3, anchor = "c")
        
        #get date time
        now = datetime.now()
        datetime_str = now.strftime("%d/%m/%y %H:%M")
        
        # Imwrite
        cv2.imwrite("/home/duongnh/Documents/Project1/Database/save_frame/"\
                        + str(self.count) + '_image.jpg', self.frame_save)
        
        if (len(self.bboxes_yolo) != 0):
            cv2.imwrite("/home/duongnh/Documents/Project1/Database/save_result/"\
                                 + str(self.count) + '_image.jpg',self.image_result_yolo)
            
            file = open("/home/duongnh/Documents/Project1/Database/save_lpname/" + str(self.count) + "_image.txt", "w")
            
            string_imshow = ""
            
            self.get_lp_recognition()
            
            for lp_name in self.lp_recognition:
                print(type(datetime_str), type(lp_name))
                if datetime_str != "" and lp_name != None:
                    file.write("Time: " + datetime_str + " Result: " + lp_name +  "\n")
                    string_imshow += lp_name + " " + "\n"
                # datetime_in = querry_t(lp_name, datetime_str)
                file.close()
                break
            
            #querry database
            result = self.database.querry_Plate_OUT(self.lp_recognition[0], datetime_str)
            if (result == None):
                string_imshow += "\n Time out: " + datetime_str + "\n"
                string_imshow += "NOT FOUND LICENSE PLATE IN DATABASE"
            else:
                string_imshow += "\n Time out: " + datetime_str + "\n"
                string_imshow += "FOUND LICENSE PLATE \n TIME IN: " + str(result)
            
            #lable imshow lpname
            myFont = font.Font(family="Helvetica",size=18,weight="bold")
            var_lp = StringVar()
            self.label_lp = Label(self.window, textvariable=var_lp , width = 40, height = 12, font =  myFont)
            var_lp.set(string_imshow)
            self.label_lp.place(relx = 0.8, rely = 0.8, anchor = "c")
        else:   
            myFont = font.Font(family="Helvetica",size=18,weight="bold")
            var_error = StringVar()
            self.label_lp = Label(self.window, textvariable=var_error , width = 40, height = 12, font =  myFont)
            var_error.set("No license plate in image")
            self.label_lp.place(relx = 0.8, rely = 0.8, anchor = "c")
            
    def thread_VehicleIN(self):
        thread = Thread(target = self.Result_VehicleIN)
        thread.start()
        
    def thread_VehicleOUT(self):
        thread = Thread(target = self.Result_VehicleOUT)
        thread.start()
    
    def create_button(self):
        button_in = Button(self.window,text = "Vehicle IN",\
                command=self.thread_VehicleIN, height = 7, width = 14,\
                    activebackground = "cyan")
        button_out = Button(self.window, text = "Vehicle OUT",\
            command = self.thread_VehicleOUT, height = 7, width = 14,\
                    activebackground = "cyan")
        
        #button.grid(row=0, column=0)
        button_in.place(relx=0.4, rely=0.9, anchor="c")
        #button.pack()
        button_out.place(relx=0.6, rely=0.9, anchor="c")
    
    
    def main(self):
        self.update_frame()
        self.window.mainloop()
        



if __name__ == "__main__":
    display = Display()
    display.main()
import requests
import cv2
import numpy as np
import time
url = "http://192.168.1.7:8080/shot.jpg"
cv2.namedWindow('image')
cv2.resizeWindow('image', 600,600)
while True:
    r = requests.get(url)
    print("Connect")
    img_arr = np.array(bytearray(r.content),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    img = cv2.resize(img, dsize = None, fx = 0.7, fy = 0.7)
    cv2.imshow('image',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
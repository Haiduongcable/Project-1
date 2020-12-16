import cv2 
import threading, queue




class Camera: 
    def __init__(self, size_queue, fps, queue_image):
        self.size_queue = size_queue
        self.fps = fps 
        self.queue_image = queue_image
        self.window_name = "Imshow Image"
    
    def get_camera_image(self):
        # Open webcam and put image to queue
        vid = cv2.VideoCapture(0) 
    
        while(True): 
            
            # Capture the video frame 
            # by frame 
            ret, frame = vid.read()
            if self.queue_image.qsize() <= self.size_queue:
                self.queue_image.put(frame)
            else:
                self.queue_image.get()
                self.queue_image.put(frame)
            
        
        # After the loop release the cap object 
        vid.release()
    
    def imshow_image(self):
        while True:
            if (not self.queue_image.empty()):
                image = self.queue_image.get()
                cv2.imshow(self.window_name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cv2.destroyAllWindows() 
    
    
if __name__ == '__main__':
    q = queue.Queue()
    thread1 = threading.Thread(target=get_camera_image,args=(q,))
    thread2 = threading.Thread(target=imshow_image,args=(q,'Thread2',))
    #thread3 = threading.Thread(target=imshow_image,args=(q,'Thread3',))
    thread1.start()
    thread2.start()
    #thread3.start()
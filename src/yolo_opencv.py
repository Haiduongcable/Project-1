import cv2
import numpy as np




class Yolo_opencv:
    def __init__(self):
        self.pathweight = "/home/duongnh/Documents/TPA/checkpoint/yolov3-tiny_3l_last.weights"
        self.pathcfg = "/home/duongnh/Documents/TPA/checkpoint/yolov3-tiny_3l.cfg"
        self.pathnames = "/home/duongnh/Documents/TPA/checkpoint/tpa.names"
        self.net = cv2.dnn.readNet(self.pathweight, self.pathcfg)
        self.classes = []
        with open("/home/duongnh/Documents/TPA/checkpoint/tpa.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # detect 
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        bboxes_nms = []
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                bboxes_nms.append(boxes[i])
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = (0,0,255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        return img, bboxes_nms
        
if __name__ == "__main__":
    detector = Yolo_opencv()
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
        image = detector.detect(frame)
        cv2.imshow("Output", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
    cv2.destroyAllWindows() 
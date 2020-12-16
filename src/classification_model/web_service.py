from flask import Flask, Response, url_for, render_template
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from multiprocessing import Process, Queue
import threading
import cv2

class WebService(Process):
    def __init__(self, frame_dict, devices_info, info_to_board, cfg):
        super(WebService, self).__init__()
        self.app = Flask(__name__)
        cors = CORS(self.app)
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.frame_dict = frame_dict
        self.cfg = cfg
        self.devices_info = devices_info
        self.info_to_board = info_to_board
        
        @self.app.route('/get_stream/<int:deviceID>', methods=['GET', 'POST'])
        @cross_origin()
        def get_stream(deviceID):
            # deviceID = request.form['deviceID']
            def gen_frame(deviceID):
                while True:
                    if deviceID not in self.frame_dict.keys():
                        # frame = cv2.imread('no-signal.jpg')
                        continue
                    else:
                        frame = self.frame_dict[deviceID][:, :, ::-1]
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            return Response(gen_frame(deviceID), mimetype='multipart/x-mixed-replace; boundary=frame')            
        
        @self.app.route('/devices', methods=['GET', 'POST'])
        @cross_origin()
        def get_devices():
            if request.method == 'GET':
                return jsonify(self.devices_info._getvalue())
            elif request.method == 'POST':
                message = "Success"
                request_info = request.json
                self.info_to_board.put(request_info)
                return jsonify({ "message": message })
            
    def run(self):
        self.app.run(host=self.cfg.ip, port=self.cfg.web_service.port, threaded=True)

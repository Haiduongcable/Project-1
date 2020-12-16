from multiprocessing import Process, Queue
import threading
import time

class FrameUpdater(Process):
    def __init__(self, current_frame, results, stop_dict):
        super(FrameUpdater, self).__init__()
        self.results = results
        self.current_frame = current_frame
        self.stop_dict = stop_dict

    def run(self):
        while not self.stop_dict["FrameUpdater"]:
            while self.results.qsize() > 10:
                result = self.results.get()
            if self.results.qsize() > 0:
                result = self.results.get()
                self.current_frame[result["deviceID"]] = result["image"]
                time.sleep(0.03)

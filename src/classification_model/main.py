import os
import sys
sys.path.append('../..')
from communicating import CommunicationService
from slave_service import ClassifyService
from multiprocessing import Process, Queue, Manager
import anyconfig
import munch
# from model import CharacterRecognition
import numpy as np
from web_service import WebService
from frame_updater import FrameUpdater
import argparse
import sys



def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--config_path', type=str, help='path to config path', default='config.yaml')
    parser.add_argument('--gpu', type=str, help='choose gpu (0,1,2,...) or cpu(-1)', default='0')
    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = anyconfig.load(args.config_path)
    cfg = munch.munchify(cfg)

    # LOAD MODEL

    manager = Manager()
    stop_dict = manager.dict()
    stop_dict["CommunicationService"] = False
    stop_dict["ClassifyService"] = False
    stop_dict["FrameUpdater"] = False

    results = Queue(20)
    devices_info = manager.list()
    frame_dict = manager.dict()     
    info_to_board = Queue(20)
    input_data_queue = Queue(20)

    frame_updater = FrameUpdater(frame_dict, results, stop_dict)   
    web_service = WebService(frame_dict, devices_info, info_to_board, cfg)
    classify_service = ClassifyService(input_data_queue, results, stop_dict, cfg)
    print("Memory used for classify service: ",sys.getsizeof(classify_service))
    communication_service = CommunicationService(input_data_queue, devices_info, info_to_board, stop_dict, cfg)

    frame_updater.daemon = True
    communication_service.daemon = True
    web_service.daemon = True
    classify_service.daemon = True
    frame_updater.start()
    communication_service.start()
    classify_service.start()
    web_service.start()
    input("======>>> Press Enter to stop <<<======")
    for k in stop_dict.keys():
        stop_dict[k] = True
    sys.exit()
    print("Stopping CommunicationService")
    communication_service.join()
    print("Stopping FrameUpdater")
    frame_updater.join()
    print("Stopping WebService")
    web_service.terminate()
    web_service.join()
    print("ClassifyService")
    classify_service.join()

if __name__ == "__main__":
    main()

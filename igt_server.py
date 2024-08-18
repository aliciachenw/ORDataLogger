import pyigtl
import numpy as np
import time
import argparse
from NDILoggerPython.NDILogger import NDITrackingWrapper
from VideoLoggerPython.videoLogger import VideoRecordWrapper
from AtracsysLoggerPython.atracsysLogger import SpryTrackTrackingWrapper
import os
import os.path
import threading
import cv2
import time
import json
# import multiprocessing ## can have some memory issue with displaying
import argparse

if __name__ == '__main__':


    #------------------------<Creating GUI>-----------------------
    #Creates the GUI
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='none', choices=['ndi', 'sprytrack', 'none'])
    parser.add_argument('--camera', type=int, default=-1)
    parser.add_argument('--port', type=int, default=18944)
    parser.add_argument('--config', type=str, default='')
    args = parser.parse_args()

    if args.camera == -1:
        record_video_flag = False
    else:
        record_video_flag = True
        video_port = args.camera
    if record_video_flag:
        video_logger = VideoRecordWrapper(video_port)

    # init
    if args.tracker.lower() == 'ndi':
        tracker_logger = NDITrackingWrapper()
        record_tracker_flag = True
    elif args.tracker.lower() == 'sprytrack':
        tracker_logger = SpryTrackTrackingWrapper()
        record_tracker_flag = True
    elif args.tracker.lower() == 'none':
        record_tracker_flag = False

    if args.config:
        config_path = args.config
        with open(config_path, 'r') as f:
            config = json.load(f)
            if "crop_rectangle_origin" in config:
                org_x, org_y = config["crop_rectangle_origin"]
                size_w, size_h = config["crop_rectangle_size"]
            else:
                org_x, org_y, size_w, size_h = None, None, None, None
            if "reshape_rectangle_size" in config:
                reshape_w, reshape_h = config["reshape_rectangle_size"]
            else:
                reshape_w, reshape_h = None, None
            if "ImToProbe" in config:
                ImToProbe = config["ImToProbe"]
                ImToProbe = np.array(ImToProbe).reshape(4, 4)
            else:
                ImToProbe = None

    if record_video_flag:
        video_logger.start_recording()
    if record_tracker_flag:
        tracker_logger.start_recording()

    if record_tracker_flag:
        tracker_thread = threading.Thread(target=tracker_logger.recording, args=(), daemon=True)

    if record_video_flag:
        video_thread = threading.Thread(target=video_logger.capture, args=(), daemon=True)
    
    if record_tracker_flag:
        tracker_thread.start()
    if record_video_flag:
        video_thread.start()

    ######### create IGT server
    server = pyigtl.OpenIGTLinkServer(port=args.port, local_server=True)

    freq = 1 / 30.0
    while True:
        if not server.is_connected():
            print("Waiting for client connection...")
            time.sleep(freq)
            continue
        
        print("Connected to client!")

        # Generate image
        if record_video_flag:
            image = video_logger.frame
            if org_x is not None:
                image = image[org_x:org_x+size_w, org_y:org_y+size_h]
            if reshape_w is not None:
                image = cv2.resize(image, (reshape_w, reshape_h))
            image = image.transpose(2, 0, 1) # HxWxC -> CxHxW
            image_message = pyigtl.ImageMessage(image, device_name="Image")
            server.send_message(image_message)
        if record_tracker_flag:
            data = tracker_logger.transform # 
            timesatmp = data[1]
            translation = np.array(data[3:6])
            rotation = np.array(data[6:15])
            matrix = np.eye(4)
            matrix[:3, :3] = rotation.reshape(3, 3)
            matrix[:3, 3] = translation
            transform_message = pyigtl.TransformMessage(matrix, device_name="ProbeToTracker", timestamp=image_message.timestamp)
            server.send_message(transform_message)
            if ImToProbe is not None:
                img_matrix = matrix @ ImToProbe
                img_transform_message = pyigtl.TransformMessage(matrix, device_name="ImageToTracker", timestamp=image_message.timestamp)
                server.send_message(img_transform_message)


        # Print received messages
        messages = server.get_latest_messages()
        for message in messages:
            print(message.device_name)

        # Do not flood the message queue,
        # but allow a little time for background network transfer
        time.sleep(freq)

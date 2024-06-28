'''
Description: Code for an NDI data logger GUI. Reads in tool(s) pose in quaternion or rotation/translation and timestamps.
Author: Alexandre Banks (Modified by Randy Moore)
Date: April 08, 2024
'''
from NDILoggerPython.NDILogger import NDITrackingWrapper
from VideoLoggerPython.videoLogger import VideoRecordWrapper
from AtracsysLoggerPython.atracsysLogger import SpryTrackTrackingWrapper
import os
import os.path
import threading
import cv2
import time
# import multiprocessing ## can have some memory issue with displaying
import argparse

def dummy_thread_func():
    while True:
        print("lollollol", end='\r')
        time.sleep(1/30)

if __name__ == '__main__':
    #------------------------<Creating GUI>-----------------------
    #Creates the GUI
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str)
    parser.add_argument('--camera', type=int, default=0)

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


    if record_video_flag:
        video_logger.start_recording()
    if record_tracker_flag:
        tracker_logger.start_recording()

    if record_tracker_flag:
        tracker_thread = threading.Thread(target=tracker_logger.recording, args=(), daemon=True)
    else:
        tracker_thread = threading.Thread(target=dummy_thread_func, args=(), daemon=True)

    if record_video_flag:
        video_thread = threading.Thread(target=video_logger.capture, args=(), daemon=True)
        display_thread = threading.Thread(target=video_logger.display, args=(), daemon=True)


    tracker_thread.start()
    if record_video_flag:
        video_thread.start()
        display_thread.start()

    if not record_video_flag:
        while True:
            time.sleep(0.5)

    else:
        while not video_logger.finish:
            if video_logger.finish:
                tracker_thread.join()
                video_thread.join()
                display_thread.join()
                if record_tracker_flag:
                    tracker_thread.end_recording()
                exit()
            time.sleep(0.5)

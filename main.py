'''
Description: Code for an NDI data logger GUI. Reads in tool(s) pose in quaternion or rotation/translation and timestamps.
Author: Alexandre Banks (Modified by Randy Moore)
Date: April 08, 2024
'''
from NDILoggerPython.NDILogger import NDITrackingWrapper
from VideoLoggerPython.videoLogger import VideoRecordWrapper
import os
import os.path
import threading
import cv2
import time
# import multiprocessing ## can have some memory issue with displaying

def dummy_thread_func():
    while True:
        print("lollollol")
        time.sleep(0.3)

if __name__ == '__main__':
    #------------------------<Creating GUI>-----------------------
    #Creates the GUI

    # ndi_logger = NDITrackingWrapper()
    video_logger = VideoRecordWrapper()

    # init
    # ndi_logger.start_recording()
    video_logger.start_recording()

    # ndi_thread = threading.Thread(target=ndi_logger.recording, args=(), daemon=True)
    ndi_thread = threading.Thread(target=dummy_thread_func, args=(), daemon=True)
    video_thread = threading.Thread(target=video_logger.capture, args=(), daemon=True)
    display_thread = threading.Thread(target=video_logger.display, args=(), daemon=True)


    ndi_thread.start()
    video_thread.start()
    display_thread.start()

    while not video_logger.finish:
        if video_logger.finish:
            
            ndi_thread.join()
            video_thread.join()
            display_thread.join()
            # ndi_logger.end_recording()
            exit()
        time.sleep(0.5)

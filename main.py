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

record_video_flag = True
dummy_ndi_flag = True

def dummy_thread_func():
    while True:
        print("lollollol", end='\r')
        time.sleep(1/30)

if __name__ == '__main__':
    #------------------------<Creating GUI>-----------------------
    #Creates the GUI
    
    if not dummy_ndi_flag:
        ndi_logger = NDITrackingWrapper()
    if record_video_flag:
        video_logger = VideoRecordWrapper()

    # init
    if not dummy_ndi_flag:
        ndi_logger.start_recording()
    if record_video_flag:
        video_logger.start_recording()

    if not dummy_ndi_flag:
        ndi_thread = threading.Thread(target=ndi_logger.recording, args=(), daemon=True)
    else:
        ndi_thread = threading.Thread(target=dummy_thread_func, args=(), daemon=True)

    if record_video_flag:
        video_thread = threading.Thread(target=video_logger.capture, args=(), daemon=True)
        display_thread = threading.Thread(target=video_logger.display, args=(), daemon=True)


    ndi_thread.start()
    if record_video_flag:
        video_thread.start()
        display_thread.start()

    if record_video_flag:
        while True:
            time.sleep(0.5)

    else:
        while not video_logger.finish:
            if video_logger.finish:
                
                ndi_thread.join()
                video_thread.join()
                display_thread.join()
                if not dummy_ndi_flag:
                    ndi_logger.end_recording()
                exit()
            time.sleep(0.5)

'''
Description: Code for an NDI data logger GUI. Reads in tool(s) pose in quaternion or rotation/translation and timestamps.
Author: Alexandre Banks (Modified by Randy Moore)
Date: April 08, 2024
'''
from NDILoggerPython.NDILogger import NDITrackingWrapper
from VideoLoggerPython.videoLogger import VideoRecordWrapper
from AtracsysLoggerPython.atracsysLogger import SpryTrackTrackingWrapper
import os.path
import threading
import time
# import multiprocessing ## can have some memory issue with displaying
import argparse


if __name__ == '__main__':
    #------------------------<Creating GUI>-----------------------
    #Creates the GUI
    
    # init
    ndi_tracker_logger = NDITrackingWrapper('ndi_tracker_calib.csv', use_thread=False)
    sprytrack_logger = SpryTrackTrackingWrapper('sprytrack_calib.csv', use_thread=False)


    ndi_tracker_logger.start_recording()
    sprytrack_logger.start_recording()


    for i in range(300):
        ndi_tracker_logger.recording()
        sprytrack_logger.recording()
        time.sleep(1/30)

    print("End recording")
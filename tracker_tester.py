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
    ndi_tracker_logger = NDITrackingWrapper('ndi_tracker.csv')
    sprytrack_logger = SpryTrackTrackingWrapper('sprytrack.csv')


    ndi_tracker_logger.start_recording()
    sprytrack_logger.start_recording()

    ndi_tracker_thread = threading.Thread(target=ndi_tracker_logger.recording, args=(), daemon=True)
    sprytrack_thread = threading.Thread(target=sprytrack_logger.recording, args=(), daemon=True)

    ndi_tracker_thread.start()
    sprytrack_thread.start()

    time.sleep(10) # run for 10 seconds
    
    ndi_tracker_thread.join()
    sprytrack_thread.join()
    ndi_tracker_logger.end_recording()
    sprytrack_logger.end_recording()

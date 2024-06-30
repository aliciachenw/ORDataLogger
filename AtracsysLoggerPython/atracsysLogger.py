'''
Description: Code for an NDI data logger GUI. Reads in tool(s) pose in quaternion or rotation/translation and timestamps.
Author: Alexandre Banks (Modified by Randy Moore)
Date: April 08, 2024
'''
#----------------------<Module Imports>------------------------
from tabnanny import check
import atracsys.stk as tracking_sdk
import time
import csv
from datetime import datetime #Module used to store system datetime
import tkinter as tk #Module used for GUI
import os
import os.path
import numpy as np
#--------------------<Setting Parameters>----------------------

USE_QUATERNIONS=False #Using quaternions by default
SAMPLE_RATE=30 #Sampling rate for NDI frames

# SAMPLE_PERIOD=int(round((1/30)*1000))
SAMPLE_PERIOD = 1.0 / SAMPLE_RATE


#Get current date/time
def get_filename():
    now = datetime.now()
    #Converts to dd-mm-YY_H-M-S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    #Data files will be saved as customname_{dt_string}
    csv_filepath = "Tracking_" #Default Name
    csv_filepath = csv_filepath + dt_string + ".csv"
    return csv_filepath

def get_geo_filepath():
    ROM_FILEPATH="D:/Wanwen/TORS/ORDataLogger/AtracsysLoggerPython/resources/cross_probe.ini" 

    return ROM_FILEPATH

# Replace atracsys.stk with atracsys.ftk for the fusionTrack.
def exit_with_error(error, tracking_system):
    print(error)
    answer = tracking_system.get_last_error()
    if answer[0] == tracking_sdk.Status.Ok:
        errors_dict = answer[1]
        for level in ['errors', 'warnings', 'messages']:
            if level in errors_dict:
                print(errors_dict[level])
    exit(1)


#-------------------<Function Definitions>---------------------

class SpryTrackTrackingWrapper():

    def __init__(self):
        self.csv_filepath = get_filename()
        self.geo_filepath = get_geo_filepath()
        self.use_quaternions = USE_QUATERNIONS
        self.tracking = False

    def init_csv(self):
        #Creates the csv file and the headings
        with open(self.csv_filepath, 'w', newline='') as file_object:
            writer_object=csv.writer(file_object)
            if self.use_quaternions: #Using quaternion format for the tool
                #CSV Header
                writer_object.writerow(["Tool ID","Timestamp","Frame #","Q0","Qx","Qy","Qz", "Tx","Ty","Tz", "Tracking Quality"])
            else: #Using rotation/translation format
                #CSV Header
                writer_object.writerow(["Tool ID","Timestamp","Frame #","Tx","Ty","Tz","R00","R01","R02","R10","R11","R12","R20","R21","R22","Tracking Quality"])
            file_object.close()
        
    def save_dat(self, data_formated):
        #NDI dat is a list of lists with:

    
        with open(self.csv_filepath,'a',newline='') as file_object:
            writer_object=csv.writer(file_object)
            writer_object.writerow(data_formated)
            file_object.close()
        print(data_formated, end='\r')
            
        
    def start_recording(self):
        self.init_csv()
        self.tracker = tracking_sdk.TrackingSystem() 
        
        if self.tracker.initialise() != tracking_sdk.Status.Ok:
            exit_with_error(
                "Error, can't initialise the atracsys SDK api.", self.tracker)
        if self.tracker.enumerate_devices() != tracking_sdk.Status.Ok:
            exit_with_error("Error, can't enumerate devices.", self.tracker)
        self.frame = tracking_sdk.FrameData()
        if self.tracker.create_frame(False, 10, 20, 20, 10) != tracking_sdk.Status.Ok:
            exit_with_error("Error, can't create frame object.",self.tracker)

        answer = self.tracker.get_enumerated_devices()
        if answer[0] != tracking_sdk.Status.Ok:
            exit_with_error("Error, can't get list of enumerated devices", self.tracker)

        print("Tracker with serial ID {0} detected".format(
            hex(self.tracker.get_enumerated_devices()[1][0].serial_number)))

        answer = self.tracker.get_data_option("Data Directory")
        if answer[0] != tracking_sdk.Status.Ok:
            exit_with_error("Error, can't read 'Data Directory' option", self.tracker)

        geometry_path = answer[1]

        for geometry in [self.geo_filepath]:
            if self.tracker.set_geometry(os.path.join(geometry_path, geometry)) != tracking_sdk.Status.Ok:
                exit_with_error("Error, can't create frame object.", self.tracker)

        self.tracking = True
        self.frame_num = 0
        
    def stop_recording(self):
        self.tracking = False
        self.tracker.close()
        
    def recording(self):
        #Setting Up NDI Device and Tracking
        while self.tracking:
            self.tracker.get_last_frame(self.frame)
            data_formated = []
            timestamp = datetime.now().timestamp()
            self.frame_num += 1
            for i, marker in enumerate(self.frame.markers):
                tool_id = marker.geometry_id
                marker_dat = [marker.position[0], marker.position[1], marker.position[2],\
                              marker.rotation[0][0], marker.rotation[0][1], marker.rotation[0][2],\
                                 marker.rotation[1][0],marker.rotation[1][1], marker.rotation[1][2],
                                  marker.rotation[2][0], marker.rotation[2][1], marker.rotation[2][2] ]
                data_formated += [tool_id, timestamp, self.frame_num] + marker_dat + [1]
            if len(data_formated) == 0:
                data_formated = [0, timestamp, self.frame_num, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0]
            self.save_dat(data_formated)
            time.sleep(SAMPLE_PERIOD)
     
'''
Description: Code for an NDI data logger GUI. Reads in tool(s) pose in quaternion or rotation/translation and timestamps.
Author: Alexandre Banks (Modified by Randy Moore)
Date: April 08, 2024
'''
#----------------------<Module Imports>------------------------
from tabnanny import check
from sksurgerynditracker.nditracker import NDITracker
import time
import csv
from datetime import datetime #Module used to store system datetime
import tkinter as tk #Module used for GUI
import os
import os.path
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

def get_rom_filepath():
    ROM_FILEPATH="NDILoggerPython/resources/HoloTORSUS_v2.rom" #Default .rom filepath

    return ROM_FILEPATH

#-------------------<Function Definitions>---------------------

class NDITrackingWrapper():

    def __init__(self):
        self.csv_filepath = get_filename()
        self.rom_filepath = get_rom_filepath()
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
        
    def save_dat(self, NDI_dat):
        #NDI dat is a list of lists with:
        #device_ID,time_stamp,frame_number,data,tracking_quality
        '''
        data : list of 4x4 tracking matrices, rotation and position,
                or if USE_QUATERNIONS is true, a list of tracking quaternions,
                column 0-2 is x,y,z column 3-6 is the rotation as a quaternion.
        
        '''
        #Formatting data for csv
        ID_list = NDI_dat[0] 
        timestamp_list = NDI_dat[1]
        frame_num_list = NDI_dat[2]
        data_list = NDI_dat[3]
        #data_list=np.array(NDI_dat[3])
        #data_list=data_list.tolist()
        qual_list = NDI_dat[4]
        num_tools = len(ID_list) #Number of tools 
        
        for i in range(num_tools): #Loops for the number of tools
            if self.use_quaternions: #Formats data in csv as if using quaternions
                new_dat=data_list[0][i].tolist()
                data_formated=[ID_list[i],timestamp_list[i],frame_num_list[i]] #,new_dat,qual_list[i]]    
                data_formated=data_formated+new_dat
                data_formated.append(qual_list[i])
            else: #Formats it with translation/rotation format
                new_dat=[data_list[i][0][3],data_list[i][1][3],data_list[i][2][3], 
                        data_list[i][0][0],data_list[i][0][1],data_list[i][0][2],
                        data_list[i][1][0],data_list[i][1][1],data_list[i][1][2],
                        data_list[i][2][0],data_list[i][2][1],data_list[i][2][2]]
                data_formated=[ID_list[i],timestamp_list[i],frame_num_list[i]] #,new_dat,qual_list[i]]    
                data_formated=data_formated+new_dat
                data_formated.append(qual_list[i])

        
            with open(self.csv_filepath,'a',newline='') as file_object:
                writer_object=csv.writer(file_object)
                writer_object.writerow(data_formated)
                file_object.close()
            print(data_formated, end='\r')
            self.transform = data_formated
            
    # def update_quaternion(self):
    #     self.use_quaternions = True
    # def update_rotation(self):
    #     self.use_quaternions = False

    def get_transform(self):
        return self.transform
        
    def start_recording(self):
        self.init_csv()

        self.settings={
                    "tracker type": "polaris",
                    "romfiles": [self.rom_filepath]
                }
        self.tracker=NDITracker(self.settings) #Sets the NDITracker object
        self.tracker.use_quaternions=self.use_quaternions #API will record data (in "tracking") as quaternions
                                            #columns 0-2: x,y,z and column 3-6:rotation as a quaternion

        self.tracker.start_tracking() #Starts tracking
        self.tracking = True
        
    def stop_recording(self):
        self.tracking = False
        self.tracker.stop_tracking()
        self.tracker.close()
        
    def recording(self):
        #Setting Up NDI Device and Tracking
        while self.tracking:
            NDI_dat=self.tracker.get_frame()
            self.save_dat(NDI_dat)
            time.sleep(SAMPLE_PERIOD)
     
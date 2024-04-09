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
#Get current date/time
now=datetime.now()
#Converts to dd-mm-YY_H-M-S
dt_string=now.strftime("%d-%m-%Y_%H-%M-%S")
#Data files will be saved as customname_{dt_string}
CSV_FILEPATH="../ParticipantData" #Default Name
CSV_FILEPATH=CSV_FILEPATH+dt_string+".csv"

ROM_FILEPATH="../resources/ecm.rom" #Default .rom filepath

USE_QUATERNIONS=True #Using quaternions by default
SAMPLE_RATE=30 #Sampling rate for NDI frames

SAMPLE_PERIOD=int(round((1/30)*1000))
TRACKING=False

#-------------------<Function Definitions>---------------------
def init_csv():
#Creates the csv file and the headings
    with open(CSV_FILEPATH,'w',newline='') as file_object:
        writer_object=csv.writer(file_object)
        if USE_QUATERNIONS: #Using quaternion format for the tool
            #CSV Header
            writer_object.writerow(["Tool ID","Timestamp","Frame #","Tx","Ty","Tz","Q0","Qx","Qy","Qz","Tracking Quality"])
        else: #Using rotation/translation format
            #CSV Header
            writer_object.writerow(["Tool ID","Timestamp","Frame #","Tx","Ty","Tz","R00","R01","R02","R10","R11","R12","R20","R21","R22","Tracking Quality"])
        file_object.close()
        
def save_dat(NDI_dat):
    #NDI dat is a list of lists with:
    #device_ID,time_stamp,frame_number,data,tracking_quality
    '''
    data : list of 4x4 tracking matrices, rotation and position,
            or if USE_QUATERNIONS is true, a list of tracking quaternions,
            column 0-2 is x,y,z column 3-6 is the rotation as a quaternion.
    
    '''
    #Formatting data for csv
    ID_list=NDI_dat[0] 
    timestamp_list=NDI_dat[1]
    frame_num_list=NDI_dat[2]
    data_list=NDI_dat[3]
    #data_list=np.array(NDI_dat[3])
    #data_list=data_list.tolist()
    qual_list=NDI_dat[4]
    num_tools=len(ID_list) #Number of tools 
    
    for i in range(num_tools): #Loops for the number of tools
        if USE_QUATERNIONS: #Formats data in csv as if using quaternions
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
    
        with open(CSV_FILEPATH,'a') as file_object:
            writer_object=csv.writer(file_object)
            writer_object.writerow(data_formated)
            file_object.close()

def update_rom():
    new_filepath=entry_rom.get()
    check_path=os.path.isfile(new_filepath)
    check_extension=new_filepath.endswith('.rom')
    if check_path and check_extension: #File exists
        global ROM_FILEPATH
        ROM_FILEPATH=new_filepath
    else:
        text_errors.insert(tk.END,"\nFile Does not Exist or wrong extension (use .rom)")
    
def update_csv():
    new_filepath=entry_csv.get()
    split_path=os.path.split(new_filepath)
    check_path=os.path.isdir(split_path[0]) #Only checks that the directory exists
    check_extension=new_filepath.endswith('.csv')
    if check_path and check_extension:
        global CSV_FILEPATH
        CSV_FILEPATH=new_filepath
    else:
        text_errors.insert(tk.END,"\nCSV Directory Does not Exist or wrong extension (use .csv)")
    
def update_quaternion():
    global USE_QUATERNIONS
    USE_QUATERNIONS=True
def update_rotation():
    global USE_QUATERNIONS
    USE_QUATERNIONS=False
    
def start_recording():
    text_errors.insert(tk.END,"\nRecording Started")
    text_errors.insert(tk.END,"\nData Stored in: "+CSV_FILEPATH)
    init_csv()
    global settings
    global tracker
    settings={
                "tracker type": "polaris",
                "romfiles": [ROM_FILEPATH]
            }
    tracker=NDITracker(settings) #Sets the NDITracker object
    tracker.use_quaternions=USE_QUATERNIONS #API will record data (in "tracking") as quaternions
                                        #columns 0-2: x,y,z and column 3-6:rotation as a quaternion

    tracker.start_tracking() #Starts tracking
    global TRACKING
    TRACKING=True
    
def stop_recording():
    text_errors.insert(tk.END,"\nRecording Stoped")
    global TRACKING
    TRACKING=False
    if 'tracker' in globals():
        tracker.stop_tracking()
        tracker.close()
    
def recording():
    #Setting Up NDI Device and Tracking
    if TRACKING:
        NDI_dat=tracker.get_frame()
        save_dat(NDI_dat)
    window.after(SAMPLE_PERIOD,recording)
    


#------------------------<Creating GUI>-----------------------
#Creates the GUI
window=tk.Tk() #Creates window
window.title("NDI Data Logger")

window.rowconfigure([0,1,2,3,4],weight=1)
window.columnconfigure([0,1,2],weight=1)

#---First Row
label_rom=tk.Label(text="1. Enter full Rom filepath/filename.rom:\n(to change default)", width=35)
label_rom.grid(row=0,column=0,sticky="nsew")

#Entry to get file path
entry_rom=tk.Entry(width=50)
entry_rom.grid(row=0,column=1,sticky="nsew")


#Button to enter file path
button_rom=tk.Button(text="Click Enter",width=15,command=update_rom)
button_rom.grid(row=0,column=2,sticky="nsew")

#---Second Row:
label_csv=tk.Label(text="2. Enter full csv filepath/filename.csv:\n(to change default)", width=35)
label_csv.grid(row=1,column=0,sticky="nsew")

#Entry to get file path
entry_csv=tk.Entry(width=50)
entry_csv.grid(row=1,column=1,sticky="nsew")


#Button to enter file path
button_csv=tk.Button(text="Enter Filename",width=15,command=update_csv)
button_csv.grid(row=1,column=2,sticky="nsew")

#---Third Row:
label_rec=tk.Label(text="3. Quaternions (default) or rot/trans?:", width=35)
label_rec.grid(row=2,column=0,sticky="nsew")

#Button to select quaternions (default)
button_quat=tk.Button(text="Quaternions", width=15,command=update_quaternion)
button_quat.grid(row=2,column=1,sticky="e")


#Button to select rotation/translation representation
button_rot=tk.Button(text="Rotation/Translation",width=15,command=update_rotation)
button_rot.grid(row=2,column=2,sticky="nsew")




#---Fourth Row: (start/stop recording)
label_rec=tk.Label(text="4. Start/Stop Recording:", width=35)
label_rec.grid(row=3,column=0,sticky="nsew")

#Button to start recording
button_start=tk.Button(text="Start", width=15,bg="green",command=start_recording)
button_start.grid(row=3,column=1,sticky="e")


#Button to stop recording
button_stop=tk.Button(text="Stop",width=15,bg="red",command=stop_recording)
button_stop.grid(row=3,column=2,sticky="nsew")


#Fifth Row (error messages)
label_rec=tk.Label(text="Messages:", width=35)
label_rec.grid(row=4,column=0,sticky="nsew")


text_errors=tk.Text(height=5)
text_errors.grid(row=4,column=1,sticky="nsew")


window.after(SAMPLE_PERIOD,recording)

window.mainloop()
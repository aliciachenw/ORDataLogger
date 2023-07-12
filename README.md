# NDI_DataLogger
Python script that runs a simple GUI to log timestamped and formatted data from an NDI Polaris tracker.

Cross-platform GUI.


# Recorded Data Format:
Timestamps data with system time. Runs at 30 Hz

Output data is a .csv file with the following header format:

Tool ID, Timestamp, Frame #, data columns, Tracking Quality


**Tool ID**: is the unique ID of each tool in the NDI Polaris field of view. (This GUI supports multiple tools)

**Timestamp**: system time

**Frame #**: the frame returned by the NDI Polaris API (**Note: use the frame number and first timestamp for synchronization.** The frame number is incremented by 1 at a constant rate of 60 Hz )

**data columns**: Either in quaternion format (Tx,Ty,Tz,q0,qx,qy,qz) or translation/rotation format (Tx,Ty,Tz,R00,R01,R02,R10,R11,R12,R20,R21,R22)

**Tracking Quality**: lists the tracking quality per tool


# Installation

$pip install -r requirements.txt



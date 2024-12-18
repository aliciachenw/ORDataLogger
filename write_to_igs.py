"""
Function to postprocess the recorded data to generate a igs.mha file
"""

import numpy as np
import SimpleITK as sitk
import cv2
import os
import argparse
from DataPostProcess.io import *
from DataPostProcess.improc import *
from DataPostProcess.timeproc import *
import json

def process(video_path, output_path, args):
    """
    Read the data from the data path
    """
    # split the video path name
    folder, video_name = os.path.split(video_path)
    video_name = video_name.split('.')[0]
    video_timeframe_path = os.path.join(folder, video_name + '.csv')
    tracking_name = video_name.replace('Recording_', 'Tracking_')
    tracking_path = os.path.join(folder, tracking_name + '.csv')
    
    print("Video path:", video_path)
    print("Timeframe path:", video_timeframe_path)
    print("Tracking path:", tracking_path)

    # load files
    frames = read_video(video_path, grayscale=True)
    # load timeframe
    _, timeframe = read_csv(video_timeframe_path)
    _, tracking = read_csv(tracking_path) # tracking: [toolID, timestamp, frame, q0, qx, qy, qz, x, y, z, quality]

    # timesync
    matchs = timesync(timeframe[:,0], tracking[:, 1], threshold=0.1)
    # print("Matchs:", matchs)
    # crop the video
    # if args.config is not None:
    #     with open(args.config, 'r') as f:
    #         config = json.load(f)
    #     frames = crop_video(frames, config['crop_rectangle_origin'][0], config['crop_rectangle_origin'][1], config['crop_rectangle_size'][0], config['crop_rectangle_size'][1])


    sequence_list = get_sequence_list(timeframe)
    # write the igs file
    # print(sequence_list)
    write_igs(frames, tracking, timeframe, matchs, output_path, sequence_list)
    # print("success write to", output_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--video_input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)

    args = argparser.parse_args()

    input_path = args.video_input
    output_path = args.output

    process(input_path, output_path, args)
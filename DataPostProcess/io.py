import cv2
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import SimpleITK as sitk


def read_video(video_path, grayscale=False):
    # load files
    print("Reading video from", video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    print("Video loaded with", len(frames), "frames")
    return frames

def read_csv(csv_path):
    # load files
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = [row for row in reader]

    # convert to float
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except:
                pass
    # convert to numpy array
    header = data[0]
    data = np.array(data[1:])
    return header, data


def write_igs(frames, transform, frame_timestamps, matches, output_path):

    # get data
    frames = frames[:100]
    print("Image size:", frames[0].shape)
    H, W = frames[0].shape
    sequence_length = len(frames)
    tracking_transform = 'ProbeToTrackerTransform'
    # get basic metadata
    curr_frame = []
    metadata = {}

    # write headers
    metadata['ObjectType'] = 'Image'
    metadata['NDims'] = '3'
    metadata['AnatomicalOrientation'] = 'RAI'
    metadata['BinaryData'] = 'True'
    metadata['BinaryDataByteOrderMSB'] = 'False'
    metadata['CenterOfRotation'] = '0 0 0'
    metadata['CompressedData'] = 'False'
    metadata['DimSize'] = str(W) + ' ' + str(H) + ' ' + str(sequence_length)
    metadata['ElementNumberOfChannels'] = '1'
    metadata['ElementSpacing'] = '1 1 1'
    metadata['ElementType'] = 'MET_UCHAR'
    metadata['Kinds'] = 'domain domain list'
    metadata['Offset'] = '0 0 0'
    metadata['TransformMatrix'] = '1 0 0 0 1 0 0 0 1'
    metadata['UltrasoundImageOrientation'] = 'MFA'
    metadata['UltrasoundImageType'] = 'BRIGHTNESS'


    for i in range(sequence_length):
        
        tracker_status = 'Seq_Frame' + str(i).zfill(4) + '_' + tracking_transform + 'Status'
        image_status = 'Seq_Frame' + str(i).zfill(4) + '_ImageStatus'
        tracking_transform = 'Seq_Frame' + str(i).zfill(4) + '_' + tracking_transform
        timestamp = 'Seq_Frame' + str(i).zfill(4) + '_Timestamp'

        if i in matches[:,0]: # have match
            idx = np.argwhere(matches[:,0] == i)[0][0]
            j = matches[idx, 1]
            if np.isnan(transform[j, 3]):
                metadata[tracker_status] = 'MISSING'
                metadata[image_status] = 'OK'
                metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
                metadata[timestamp] = str(frame_timestamps[i] - frame_timestamps[0])
            else:
                metadata[tracker_status] = 'OK'
                metadata[image_status] = 'OK'
                metadata[tracking_transform] = cvt_transform(transform[j])
                metadata[timestamp] = str(frame_timestamps[i] - frame_timestamps[0])
        
        else:
            metadata[tracker_status] = 'MISSING'
            metadata[image_status] = 'OK'
            metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
            metadata[timestamp] = str(frame_timestamps[i] - frame_timestamps[0])

    # write to file
    image_npy = np.array(frames).astype(np.uint8) # S, W, H
    image_npy = image_npy.transpose(1, 2, 0) # W, H, S
    image = sitk.GetImageFromArray(np.array(frames).astype(np.uint8))

    print("Writing image to", output_path)
    print(np.array(frames).shape)
    for key in metadata:
        image.SetMetaData(key, metadata[key])

    sitk.WriteImage(image, output_path)
    print("Image written to", output_path)


def cvt_transform(t):
    """
    Convert to matrix then convert to string
    """
    # t: toolID, timestamp, frame, q0, qx, qy, qz, x,y,z, quality

    if np.isnan(t[3]):
        return "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
    mat = np.eye(4)
    mat[0, 3] = t[7] # x
    mat[1, 3] = t[8] # y
    mat[2, 3] = t[9] # z

    rot = R.from_quat(t[3:7])
    mat[:3, :3] = rot.as_matrix()

    return ' '.join([str(x) for x in mat.flatten()])
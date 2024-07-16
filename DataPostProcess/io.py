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
            if len(frames) == 0:
                print("Image size:", frame.shape)
                cv2.imwrite("test.png", frame)
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


def write_igs(frames, transform, frame_timestamps, matches, output_path, sequence_list=None):

    if sequence_list is None:
        # get data
        # frames = frames
        print("Image size:", frames[0].shape)
        H, W = frames[0].shape
        sequence_length = len(frames)
        FIX = 'ProbeToTrackerTransform'
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
            
            tracker_status = 'Seq_Frame' + str(i).zfill(4) + '_' + FIX + 'Status'
            image_status = 'Seq_Frame' + str(i).zfill(4) + '_ImageStatus'
            tracking_transform = 'Seq_Frame' + str(i).zfill(4) + '_' + FIX
            timestamp = 'Seq_Frame' + str(i).zfill(4) + '_Timestamp'

            if i in matches[:,0]: # have match
                idx = np.argwhere(matches[:,0] == i)[0][0]
                print(idx)
                j = matches[idx, 1]
                print(transform[j])
                if np.isnan(transform[j, 3]):
                    metadata[tracker_status] = 'MISSING'
                    metadata[image_status] = 'OK'
                    metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
                    metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])
                else:
                    metadata[tracker_status] = 'OK'
                    metadata[image_status] = 'OK'
                    metadata[tracking_transform] = cvt_transform(transform[j])
                    metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])
            
            else:
                metadata[tracker_status] = 'MISSING'
                metadata[image_status] = 'OK'
                metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
                metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])

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

    else:
        
        for i, (seq_st, seq_end) in enumerate(sequence_list):
            sequence_length = seq_end - seq_st
            # get data
            print("Image size:", frames[0].shape)
            H, W = frames[0].shape
            sequence_length = len(frames)
            FIX = 'ProbeToTrackerTransform'
            # get basic metadata
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


            for kk in range(seq_st, seq_end):
                
                tracker_status = 'Seq_Frame' + str(kk - seq_st).zfill(4) + '_' + FIX + 'Status'
                image_status = 'Seq_Frame' + str(kk - seq_st).zfill(4) + '_ImageStatus'
                tracking_transform = 'Seq_Frame' + str(kk - seq_st).zfill(4) + '_' + FIX
                timestamp = 'Seq_Frame' + str(kk - seq_st).zfill(4) + '_Timestamp'
                # print(matches)
                # exit()
                if kk in matches[:,0]: # have match
                    idx = np.argwhere(matches[:,0] == kk)[0][0]
                    print(idx)
                    j = matches[idx, 1]
                    print(transform[j])
                    if np.isnan(transform[j, 3]):
                        metadata[tracker_status] = 'MISSING'
                        metadata[image_status] = 'OK'
                        metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
                        metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])
                    else:
                        metadata[tracker_status] = 'OK'
                        metadata[image_status] = 'OK'
                        metadata[tracking_transform] = cvt_transform(transform[j])
                        metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])
                
                else:
                    metadata[tracker_status] = 'MISSING'
                    metadata[image_status] = 'OK'
                    metadata[tracking_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
                    metadata[timestamp] = str(frame_timestamps[i,0] - frame_timestamps[0,0])

            # write to file
            image_npy = np.array(frames[seq_st:seq_end]).astype(np.uint8) # S, W, H
            # image_npy = image_npy.transpose(1, 2, 0) # W, H, S
            image = sitk.GetImageFromArray(image_npy.astype(np.uint8))


            temp_output_path = output_path.replace('.igs.mha', '_%d.igs.mha' % i)
            print("Writing image to", temp_output_path)
            print(image_npy.shape)
            for key in metadata:
                image.SetMetaData(key, metadata[key])

            sitk.WriteImage(image, temp_output_path)
            print("Image written to", temp_output_path)

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


def cvt_string_to_transform(string):
    """
    Convert to matrix then convert to string
    """
    # t: toolID, timestamp, frame, q0, qx, qy, qz, x,y,z, quality

    sub_str = string.split(' ')
    if sub_str[0] == "-nan(ind)":
        return np.array([[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 1]])

    mat = np.eye(4)
    for i in range(4):
        for j in range(4):
            mat[i, j] = float(sub_str[i*4 + j])
    # print(mat)
    return mat



def cvt_transform_mat_to_string(mat):
    string = ' '.join([str(x) for x in mat.flatten()])
    return string
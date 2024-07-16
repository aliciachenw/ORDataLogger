import numpy as np
import SimpleITK as sitk


def read_sequence(data_path, tracking_transform):

    # get meta data
    reader = sitk.ImageFileReader()
    reader.SetFileName(data_path)   # Give it the mha file as a string
    reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    reader.ReadImageInformation()  # Get just the information from the file

    # get data
    data = reader.Execute()
    # print("Image size:", data.GetSize())
    image_npy = sitk.GetArrayFromImage(data)
    print("Image size:", image_npy.shape)
    tracking_data, tracking_status = get_tranform_data(data, transform=tracking_transform)
    timestamp = get_timestamp(data)
    assert len(tracking_data) == len(tracking_status) == len(timestamp)
    print("Sequence length:", len(tracking_data), " visible frame:", np.sum(tracking_status))
    tracking_data = np.stack(tracking_data, axis=0)
    # print(tracking_data.shape)
    return {'tracking_seq':tracking_data, 
            'tracking_status': tracking_status, 
            'timestamp': timestamp, 
            'image_seq': image_npy}


def read_sequence_npy(data_path):

    # get meta data
    reader = sitk.ImageFileReader()
    reader.SetFileName(data_path)   # Give it the mha file as a string
    reader.LoadPrivateTagsOn()     # Make sure it can get all the info
    reader.ReadImageInformation()  # Get just the information from the file

    # get data
    data = reader.Execute()
    # print("Image size:", data.GetSize())
    image_npy = sitk.GetArrayFromImage(data)
    print("Image size:", image_npy.shape)

    return {'image_seq': image_npy}


def cvt_transform_sequences_from_string_to_array(sequences):
    new_sequences = []

    for i, trans_str in enumerate(sequences):
        
        trans = trans_str.split()
        if trans[0].find('ind') != -1:
            T = np.ones((4, 4)) * np.nan
        else:
            T = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    T[i][j] = float(trans[i * 4 + j])
            new_sequences.append(T)
            
    return new_sequences

def get_tranform_data(meta_data, transform='ImageToTrackerTransform'):

    transform_sequence = []
    transform_status = []
    meta_keys = meta_data.GetMetaDataKeys()
    for key in meta_keys:
        if key.find(transform) != -1 and key.find('Status') == -1:
            # print(key)
            # print(reader.GetMetaData(f'{key}'))
            transform_sequence.append(meta_data.GetMetaData(f'{key}'))
            status_key = key + 'Status'
            transform_stat = meta_data.GetMetaData(f'{status_key}')
            # print(transform_stat)
            if transform_stat == 'MISSING':
                transform_status.append(False)
                # print(False)
            elif transform_stat == 'OK':
                transform_status.append(True)
                # print(True)

    assert len(transform_status) == len(transform_sequence)
    # refine the transformation
    transform_sequence = cvt_transform_sequences_from_string_to_array(transform_sequence)
    return transform_sequence, transform_status



def get_timestamp(meta_data):
    timestamp = []
    meta_keys = meta_data.GetMetaDataKeys()
    for key in meta_keys:
        if key.find('Timestamp') != -1 and key.find('Status') == -1:
            timestamp.append(meta_data.GetMetaData(f'{key}'))
    return timestamp



if __name__ == '__main__':
    
    data_path = 'D:/Wanwen/TORS/raw_data/IPCAI_freehand_US/Output/HV004-left/record104723.igs.mha'
    data_dict1 = read_sequence(data_path, 'ImageToTrackerTransform')
    data_dict2 = read_sequence(data_path, 'ProbeToTrackerTransform')
    # print(len(data_dict['tracking_seq']))
    # print(data_dict['image_seq'].shape)

    image2probe = np.array([[-0.0147571,-0.0784815,0.0115738,-89.0484],
    [-0.0789898,0.013447,-0.00953197,39.5666],
    [0.00734219,-0.013073,-0.0792859,-36.324], 
    [0,0,0,1]
    ])

    print(data_dict1['tracking_seq'][0])
    print(data_dict2['tracking_seq'][0])
    print(data_dict2['tracking_seq'][0] @ image2probe)
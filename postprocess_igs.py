import SimpleITK as sitk
import os
import numpy as np
import cv2
import argparse
import json
from DataPostProcess.io import *
from DataPostProcess.improc import *



def main(image_path, output_path, config_path):
    # read image
    image = sitk.ReadImage(image_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # crop and reshape
    image_array = sitk.GetArrayFromImage(image)
    image_array = crop_arrays(image_array, config['crop_rectangle_origin'][0], config['crop_rectangle_origin'][1], config['crop_rectangle_size'][0], config['crop_rectangle_size'][1])
    image_array = resize_array(image_array, config['reshape_rectangle_size'])

    # add transform
    calib_mat = config["ImToProbe"]
    calib_mat = np.array(calib_mat).reshape(4, 4)

    sequence_length = image_array.shape[0]
    # get data
    print("Image size:", image_array[0].shape)
    H, W = image_array[0].shape
    FIX = 'ProbeToTrackerTransform'
    NEW_TRANSFORM = 'ImageToTrackerTransform'
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


    for i in range(sequence_length):
        
        tracker_status = 'Seq_Frame' + str(i).zfill(4) + '_' + FIX + 'Status'
        image_status = 'Seq_Frame' + str(i).zfill(4) + '_ImageStatus'
        tracking_transform = 'Seq_Frame' + str(i).zfill(4) + '_' + FIX
        timestamp = 'Seq_Frame' + str(i).zfill(4) + '_Timestamp'
        transform =  image.GetMetaData(tracking_transform)
        transform = cvt_string_to_transform(transform)

        image_transform_status = 'Seq_Frame' + str(i).zfill(4) + '_' + NEW_TRANSFORM + 'Status'
        image_transform = 'Seq_Frame' + str(i).zfill(4) + '_' + NEW_TRANSFORM

        metadata[tracker_status] = image.GetMetaData(tracker_status)
        metadata[image_status] =  image.GetMetaData(image_status)
        metadata[tracking_transform] =  image.GetMetaData(tracking_transform)
        metadata[timestamp] =  image.GetMetaData(timestamp)

        if np.any(np.isnan(transform)):
            metadata[image_transform_status] = 'MISSING'
            metadata[image_transform] = "-nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 -nan(ind) -nan(ind) -nan(ind) 0 0 0 0 1"
        else:
            metadata[image_transform_status] = 'OK'
            new_transform = transform @ calib_mat
            # print(new_transform)
            metadata[image_transform] = cvt_transform_mat_to_string(new_transform)


    new_image = sitk.GetImageFromArray(image_array.astype(np.uint8))
    print(image_array.shape)
    for key in metadata:
        new_image.SetMetaData(key, metadata[key])

    sitk.WriteImage(new_image, output_path)
    print("Image written to", output_path)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--config', type=str, required=True)

    # read image
    args = argparser.parse_args()
    input_path = args.input
    output_path = args.output
    config_path = args.config

    main(input_path, output_path, config_path)

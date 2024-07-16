from us3DReconstruction.reconstruct import Process
from us3DReconstruction.data_utils import *
import argparse
import os

import numpy as np




def reconstruct_3D_US(filename, save_path='.', auto_PCA=False):

    # read raw image file
    filename_split = os.path.split(filename)
    f = filename_split[-1][:-8]
    # folder = filename_split[0]
    # reconstruct original volume
    data_dict = read_sequence(filename, 'ImageToTrackerTransform')
    image_seq = data_dict['image_seq']
    tracking_seq = data_dict['tracking_seq']

    

    # initialize the reconstructor
    reconstruct_filter = Process()
    reconstruct_filter.init(tracking_seq, image_seq)
    reconstruct_filter.calculatePoseForUSImages()

    # Set time frames for images that can be cointaned in the voxel array
    reconstruct_filter.setValidFramesForVoxelArray(voxFrames='auto') # if use auto need to make sure input frames are all visible!

    # Calculate convenient pose for the voxel array
    #convR = T
    if auto_PCA:
        convR = 'auto_PCA'
    else:
        convR = np.eye(4)
    reconstruct_filter.calculateConvPose(convR)

    # Set scale factors -> fxyz is pixel/mm
    fxyz = (3,3,3)
    reconstruct_filter.setScaleFactors(fxyz)

    # Calculate voxel array dimensions
    reconstruct_filter.calculateVoxelArrayDimensions()

    # Allocate memory for voxel array
    reconstruct_filter.initVoxelArray()

    # Set parameters for calculating US images sequence wrapper (or silhouette) -> this is to calculate which region needs to be interpolated, seems to need this before the gap filling
    reconstruct_filter.setUSImagesAlignmentParameters(wrapper='convex_hull', step=1)

    # Align each US image of each file in the space (will take a bit ...) -> this is to cast the sequences to volume
    reconstruct_filter.alignUSImages()

    # # # Set parameters for gap filling (into the wrapped seauence)
    reconstruct_filter.setGapFillingParameters(method='VNN', blocksN=100, blockDir='X', distTh=None)

    # # # Fill gaps (go sipping a coffee ...)
    reconstruct_filter.fillGaps()

    # Set properties for the vtkImageData objects that will exported just below
    # reconstruct_filter.setImageDataProperties(sxyz=(0.1, 0.1, 0.1))
    #p.setVtkImageDataProperties(sxyz=(1,10,1)) # in case convR=T

    reconstruct_filter.writeData(save_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--auto_PCA', action='store_true') # TODO: PCA generates weird results! Need to fix

    args = argparser.parse_args()

    input_path = args.input
    output_path = args.output

    reconstruct_3D_US(input_path, output_path, args.auto_PCA)
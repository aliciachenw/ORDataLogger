import numpy as np
import os
import csv
import argparse
from DataPostProcess.io import *
from DataPostProcess.improc import *
from DataPostProcess.timeproc import *
import json


def get_calib(ndi_filename, spraytrack_filename):
    # read file
    _, ndi_tracking = read_csv(ndi_filename)
    _, spraytrack_tracking = read_csv(spraytrack_filename)

    matches = timesync(ndi_tracking[:,1], spraytrack_tracking[:,1], threshold=1/30)

    # get points
    ndi_points = []
    spraytrack_points = []

    for i in range(matches.shape[0]):
        ndi_pt = ndi_tracking[matches[i,0], 3:6]
        spraytrack_pt = spraytrack_tracking[matches[i,1], 3:6]
        # print(ndi_pt, spraytrack_pt)
        # exit()
        if np.isnan(ndi_pt).any() or np.isnan(spraytrack_pt).any():
            continue
        ndi_points.append(ndi_pt)
        spraytrack_points.append(spraytrack_pt)


    # get calibration
    ndi_points = np.array(ndi_points).T
    spraytrack_points = np.array(spraytrack_points).T
    print("NDI points shape:", ndi_points.shape)
    print("Spraytrack points shape:", spraytrack_points.shape)
    
    # estimate rigid transformation
    ####### https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    ndi_mean = np.mean(ndi_points, axis=1).reshape(-1,1)
    spraytrack_mean = np.mean(spraytrack_points, axis=1).reshape(-1,1)
    ndi_sub = ndi_points - ndi_mean
    spraytrack_sub = spraytrack_points - spraytrack_mean
    H = ndi_sub @ spraytrack_sub.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = - R @ ndi_mean + spraytrack_mean
    print("Rotation matrix:", R)
    print("Translation vector:", t)
    # recon error
    proj_ndi_to_spraytrack = R @ ndi_points  + t
    print(ndi_mean, spraytrack_mean)
    print("Projected:", proj_ndi_to_spraytrack.shape)
    error = np.linalg.norm(proj_ndi_to_spraytrack - spraytrack_points, axis=0)
    print
    print("Reconstruction error:", np.mean(error), np.std(error))

    import matplotlib.pyplot as plt

    # plot 3D motion
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(ndi_points[0], ndi_points[1], ndi_points[2], 'bo')
    ax.plot(proj_ndi_to_spraytrack[0], proj_ndi_to_spraytrack[1], proj_ndi_to_spraytrack[2], 'go')
    ax.plot(spraytrack_points[0], spraytrack_points[1], spraytrack_points[2], 'ro')
    plt.show()
    return R, t # project sprytrack to ndi



if __name__ == '__main__':
    get_calib("ndi_tracker_calib.csv", "sprytrack_calib.csv")




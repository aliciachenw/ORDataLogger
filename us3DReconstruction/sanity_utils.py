import numpy as np
from six import string_types
import os

def checkOdd(n):
    if n % 2 != 0:
        return True
    return False

def checkInt(n):
    if abs(round(n)-n) == 0:
        return True
    return False


def checkDataSequence(image_seq, tracking_seq):
    if image_seq is None:
        raise Exception('Image files were not set')
    if tracking_seq is None:
        raise Exception('Tracking files were not set')
    if image_seq.shape[0] != tracking_seq.shape[0]:
        raise Exception('Numbers of kinematics files and image files are not match!')


def checkPose(T_pose):
    if T_pose is None:
         raise Exception('Poses were not set')
    if len(T_pose.shape) != 3 or T_pose.shape[1] != 4 or T_pose.shape[2] != 4:
        raise Exception('Pose matrix must be a N x 4 x 4 matrix')


def checkGl2ConvPose(R):
    if R is None:
        raise Exception('Pose from global to convenient reference frame to was not set')
    if isinstance(R, string_types):
        if R not in ['auto_PCA','first_last_frames_centroid']:
            raise Exception(' global-to-convenient roto-translation matrix calculation method not supported')
        return
    if len(R.shape) != 2 or R.shape[0] != 4 or R.shape[1] != 4:
        raise Exception('If matrix, global-to-convenient roto-translation matrix must be 4 x 4 matrix')

def checkFxyz(fxyz):
    if fxyz is None:
        raise Exception('Voxel array scaling factors were not set')
    if isinstance(fxyz, string_types):
        if fxyz not in ['auto_bounded_parallel_scans']:
            raise Exception('Voxel array scaling factors calculation method not supported')
        return
    if len(fxyz) != 3:
        raise Exception('Voxel array scaling factors must be exactly 3')
    for i in range(0,3):
        if fxyz[i] <= 0:
            raise Exception('All voxel array scaling factors must be positive')

def checkWrapper(wrapper):
    if wrapper is None:
        raise Exception('Wrapping method was not set')
    if wrapper not in ['parallelepipedon', 'convex_hull','none']:
        raise Exception('Wrapping method not supported')

def checkStep(step):
    if step is None:
        raise Exception('Wrapping creation step was not set')
    if not checkInt(step) or step <= 0:
        raise Exception('Wrapping creation step must be integer and positive')

def checkV(V):
    if V is None:
        raise Exception('Voxel array initialization was not performed')

def checkPathForSuppFiles(fp):
    if fp is None:
        raise Exception('Path for support files was not set')
    if not os.path.isdir(fp):
        raise Exception('Path for support files is not valid')

def checkMethod(method):
    if method is None:
        raise Exception('Gaps filling method was not set')
    if method not in ['VNN', 'AVG_CUBE']:
        raise Exception('Gaps filling method not supported')

def checkBlocksN(blocksN):
    if blocksN is None:
        raise Exception('Blocks number was not set')
    if not checkInt(blocksN) or blocksN <= 0:
        raise Exception('Blocks number must be integer and positive or zero')

def checkBlockDir(d):
    if d is None:
        raise Exception('Blocks direction was not set')
    if d not in ['X', 'Y', 'Z']:
        raise Exception('Blocks direction not supported')

def checkMaxS(maxS):
    if maxS is None:
        raise Exception('Max search cube side was not set')
    if not checkInt(maxS) or not checkOdd(maxS) or maxS <= 0:
        raise Exception('Max search cube side must be integer, positive and odd')

def checkDistTh(d):
    if d is not None and d < 1:
        raise Exception('Distance threshold must be greater or equal than 1')

def checkMinPct(minPct):
    if minPct is None:
        raise Exception('Acceptability percentage was not set')
    if minPct < 0:
        raise Exception('Acceptability percentage must be positive or zero')

def checkSxyz(sxyz):
    if sxyz is None:
        raise Exception('vtkImageData spacing factors were not set')
    if isinstance(sxyz, string_types):
        if sxyz not in ['auto']:
            raise Exception('vtkImageData spacing factors calculation method not supported')
        return
    if len(sxyz) != 3:
        raise Exception('vtkImageData spacing factors must be exactly 3')
    for i in range(0,3):
        if sxyz[i] <= 0:
            raise Exception('All vtkImageData spacing factors must be positive')

def checkFilePath(p):
    if p is None:
        raise Exception('File path was not set')
    if len(p) == 0:
        raise Exception('File path cannot be empty')

def checkPrecType(p):
    if p is None:
        raise Exception('Precision type was not set')
    if p not in ['RP']:
        raise Exception('Precision type not supported')

def checkAccType(a):
    if a is None:
        raise Exception('Accuracy type was not set')
    if a not in ['DA', 'RA']:
        raise Exception('Accuracy type not supported')

def checkDist(d):
    if d is None:
        raise Exception('Distance was not set')
    if not checkInt(d) or d <= 0:
        raise Exception('Distance must be integer and positive')

def checkTimeVector(t):
    if t is None:
        raise Exception('Time vector was not set')
    if len(t) == 0:
        raise Exception('Time vector cannot be empty')
#    if t[0] != 0:
#        raise Exception('First time element must be 0')

def checkTimeDelay(t):
    if t is None:
        raise Exception('Time delay was not set')


def setInsideRange(v, bound, stepBase):
    while True:
        if v <= bound and v >= -bound:
            break
        step = -np.sign(v) * stepBase
        v += step
    return v


def checkCalibMethod(method):
    if method is None:
        raise Exception('Calibation method was not set')
    if method not in ['eq_based', 'maximize_NCCint', 'maximize_NCC', 'maximize_NCCfast']:
        raise Exception('Calibration method not supported')


def checkAlignFrames(alignFrames, N):
    if alignFrames is None:
        raise Exception('Frames for alignment were not set')
    if min(alignFrames) < 0 or max(alignFrames) > N-1:
        raise Exception('Some frame for alignment out of bounds')


def checkFillVoxMethod(method):
    if method is None:
        raise Exception('Voxel filling method was not set')
    if method not in ['last', 'avg', 'max']:
        raise Exception('Voxel filling method not supported')


def checkVoxFrames(voxFrames, N):
    if voxFrames is None:
        raise Exception('Frames for voxel array reconstruction were not set')
    if isinstance(voxFrames, string_types):
        if voxFrames not in ['all','auto']:
            pass
        return
    if min(voxFrames) < 0 or max(voxFrames) > N-1:
        raise Exception('One frame for voxel array reconstruction out of bounds')


def checkVoxFramesBounds(voxFramesBounds, N):
    if voxFramesBounds is not None:
        if voxFramesBounds[0] < 0 or voxFramesBounds[1] > N-1:
            raise Exception('Frame bounds for voxel array reconstruction lesser than 0 or bigger than %d' % N)


def checkTemporalCalibMethod(method):
    if method is None:
        raise Exception('Temporal calibration method was not set')
    if method not in ['vert_motion_sync']:
        raise Exception('Temporal calibration method not supported')
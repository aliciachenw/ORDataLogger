import cv2
import numpy as np

def crop_video(frames, x, y, w, h):
    """
    Crop the video frames
    """
    return [frame[y:y+h, x:x+w] for frame in frames]

def resize_array(arrays, new_shape):
    """
    Reshape the image
    """
    new_images = []
    for i in range(arrays.shape[0]):
        new_images.append(cv2.resize(arrays[i], new_shape))
    return np.array(new_images)

def crop_arrays(arrays, x, y, w, h):
    """
    Crop the image
    """
    return arrays[:,y:y+h, x:x+w]

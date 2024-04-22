import cv2

def crop_video(frames, x, y, w, h):
    """
    Crop the video frames
    """
    return [frame[y:y+h, x:x+w] for frame in frames]
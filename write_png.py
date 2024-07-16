import numpy as np
import cv2
def read_video(video_path, grayscale=False):
    # load files
    print("Reading video from", video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frames) == 0:
                print("Image size:", frame.shape)
            cv2.imwrite("video/frame_" + str(counter).zfill(4) + ".png", frame)
            counter += 1
        frames.append(frame)
    cap.release()
    print("Video loaded with", len(frames), "frames")
    return frames


if __name__ == '__main__':
    # read_video("D:/Wanwen/TORS/raw_data/OR_04182024/Surgery1/Recording_18_04_2024_08_23_33.avi", grayscale=True)
    cap = cv2.VideoCapture('Recording_30_06_2024_10_07_27.avi')
    frames = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(frames) == 0:
            print("Image size:", frame.shape)
        cv2.imwrite("test2.png", frame)
        counter += 1
        break
    cap.release()
    print("Video loaded with", len(frames), "frames")

import cv2
from datetime import datetime
import csv
import time

### Change the following parameters according to hardware###
videoPortIndex = 1 # --> keep modifying this integer (+1) if you do not see the video stream 
recording_framerate = 30.0
#######################################



class VideoRecordWrapper():

    def __init__(self):
        
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.filename = 'Recording_' + timestamp + '.avi'
        self.timestamp_filename = 'Recording_' + timestamp + '.csv'
        self.ret = False


        with open(self.timestamp_filename, 'w', newline='') as file_object:
            writer_object = csv.writer(file_object)
            writer_object.writerow(['Timestamp'])
            file_object.close()
        self.finish = False

    def start_recording(self):
        self.cap = cv2.VideoCapture(videoPortIndex)
        if not self.cap.isOpened():
            print("Cannot capture video input... exiting")
            exit()

        #determine size of streamed frame
        ret, frame = self.cap.read()
        self.frame = frame
        dimensions = frame.shape #returns (width, height, depth)
        height = dimensions[1]
        width = dimensions[0]

        self.out = cv2.VideoWriter(self.filename, self.fourcc, recording_framerate, (height,  width))
        print("success in capturing video, video shape:", frame.shape)


    def capture(self):
        # Capture frame-by-frame
        while not self.finish:
            try:
                self.ret, self.frame = self.cap.read()
                timestamp = datetime.now().timestamp()
                # if frame is read correctly ret is True
                if not self.ret:
                    print("Can't receive frame")
                else:
                    # Display the resulting frame
                    self.out.write(self.frame)
                    # print("receive frame:", timestamp)
                    with open(self.timestamp_filename,'a',newline='') as file_object:
                        writer_object = csv.writer(file_object)
                        writer_object.writerow([timestamp])
                        file_object.close()
            except:
                pass
            time.sleep(1 / recording_framerate)


    def end_recording(self):
        # When everything done, release the capture
        self.cap.release()
        self.out.release()
        self.finish = True
        cv2.destroyAllWindows()

    def display(self):
        while True:
            try:
                if self.ret:
                    cv2.imshow("US", self.frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                pass
        self.end_recording()

    
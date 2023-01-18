from video_realsense_file import Video
import cv2
import sys
import numpy as np
from face_detection import FaceDetection

if __name__ == '__main__':

    input_realsense = Video()
    dataPath = sys.argv[1]
    input_realsense.dataPath = dataPath
    print('init')
    print(sys.argv[1])
    fd = FaceDetection()
    input_realsense.start()
    while 1:

        rgb_frame, nir_frame = input_realsense.get_frame()
        rgb_face, rgb_face_rect = fd.face_detect_rgb(rgb_frame)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        #rgb_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
        cv2.imshow('rgb', rgb_frame)
        #cv2.imshow('rgb face', rgb_face)
        cv2.imshow('nir', nir_frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

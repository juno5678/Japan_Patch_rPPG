import numpy as np
import dlib
import cv2
import sys
from video_realsense_file import Video
import time

if __name__ == '__main__':

    input_realsense = Video()
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    path = sys.argv[1]
    input_realsense.dataPath = path
    input_realsense.start()
    frame_count = 0
    start = time.time()
    while 1:
        frame_count += 1
        rgb_frame, nir_frame = input_realsense.get_frame()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray,1)
        if (len(faces) > 0):
            for k,d in enumerate(faces):
                cv2.rectangle(rgb_frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
                #shape = landmark_predictor(gray,d)
                #for i in range(68):
                #    if i == 57 or i == 36 or i == 45:
                #        cv2.circle(rgb_frame, (shape.part(i).x, shape.part(i).y),2,(255, 0,0), -1, 3)
                #        #cv2.putText(rgb_frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                #    else :
                #        cv2.circle(rgb_frame, (shape.part(i).x, shape.part(i).y),2,(0, 255,0), -1, 3)
                #        #cv2.putText(rgb_frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

        end = time.time()
        second = end - start
        fps = frame_count / second
        second_str = 'Time taken : {:.3f} seconds '.format(second)
        fps_str = 'Estimated frames per second : {:.3f} '.format(fps)
        cv2.putText(rgb_frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.putText(rgb_frame, second_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Frame', rgb_frame)
        if (cv2.waitKey(1) == 27) or frame_count >= 300:
            print(second_str)
            print(fps_str)
            cv2.destroyAllWindows()
            break


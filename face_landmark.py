import numpy as np
import dlib
import cv2
import sys
from video_realsense_file import Video
import time
import random


# get the random weights for the height and width of the patches
def get_patch_weights():
    return [random.random(), random.random()]


# get the upper left point & size of the patch
def get_patch(weights, leftEye, rightEye, mouse):
    top_y = min(leftEye.y, rightEye.y)
    bottom_y = mouse.y
    left_x = leftEye.x
    right_x = rightEye.x
    width = right_x - left_x
    height = bottom_y - top_y
    patch_width = round(0.3*width)
    patch_height = round(0.3*height)
    patch_x = round(weights[0] * (0.7 * width) + left_x)
    patch_y = round(weights[1] * (0.7 * height) + top_y)
    return [patch_x, patch_y, patch_width, patch_height]


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
        patches_weights = get_patch_weights()
        frame_count += 1
        rgb_frame, nir_frame = input_realsense.get_frame()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 1)
        if (len(faces) > 0):
            for k, d in enumerate(faces):
                cv2.rectangle(rgb_frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
                shape = landmark_predictor(gray, d)
                leftEye = shape.part(36)
                rightEye = shape.part(45)
                mouse = shape.part(57)
                [x, y, w, h] = get_patch(patches_weights, leftEye, rightEye, mouse)
                #print(x,y,w,h)

                cv2.rectangle(rgb_frame, (x, y), (x + w, y+h), (0, 0, 255))
                cv2.circle(rgb_frame, (leftEye.x, leftEye.y), 2, (255, 0, 0), -1, 3)
                cv2.circle(rgb_frame, (rightEye.x, rightEye.y), 2, (255, 0, 0), -1, 3)
                cv2.circle(rgb_frame, (mouse.x, mouse.y), 2, (255, 0, 0), -1, 3)
                #for i in range(68):
                #    if i == 57 or i == 36 or i == 45:
                #        cv2.circle(rgb_frame, (shape.part(i).x, shape.part(i).y), 2, (255, 0, 0), -1, 3)
                        # cv2.putText(rgb_frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255))
                    # else :
                    # cv2.circle(rgb_frame, (shape.part(i).x, shape.part(i).y),2,(0, 255,0), -1, 3)
                    # cv2.putText(rgb_frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

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

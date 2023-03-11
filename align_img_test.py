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
    nir_fd = FaceDetection()
    g_fd = FaceDetection()
    input_realsense.start()
    i = 0
    total = 0
    while 1:

        #for j in range(2):
        #    rgb_frame, nir_frame = input_realsense.get_frame()

        rgb_frame, nir_frame = input_realsense.get_frame()
        g_frame = rgb_frame[:, :, 1]
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        if i == 0:
            nir_face_rects = nir_fd.face_detect(nir_frame)
        #    g_face_rects = g_fd.face_detect(g_frame)
            g_face_rects = g_fd.face_detect(gray)
        else:
            nir_face_rects = nir_fd.face_track(nir_frame)
        #    g_face_rects = g_fd.face_track(gray)
            g_face_rects = g_fd.face_track(gray)

        nir_shape = nir_fd.detect_landmark(nir_frame, nir_face_rects)
        nir_key_landmark = nir_fd.get_key_landmark(nir_shape)
        #g_shape = g_fd.detect_landmark(g_frame, g_face_rects)
        g_shape = g_fd.detect_landmark(gray, g_face_rects)
        g_key_landmark = g_fd.get_key_landmark(g_shape)
        for c in g_key_landmark:
            cv2.circle(gray, (c.x, c.y), 2, 255, -1, 3)
        for c in nir_key_landmark:
            cv2.circle(nir_frame, (c.x, c.y), 2, 255, -1, 3)
        #rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        #rgb_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
        #cv2.imshow('rgb', rgb_frame)
        #cv2.imshow('rgb face', rgb_face)
        #total += np.mean(nir_face)
        cv2.imshow('nir', nir_frame)
        cv2.imshow('g', g_frame)
        cv2.imshow('gray', gray)
        cv2.imshow('rgb', rgb_frame)
        #cv2.imshow('nir_face', nir_face)
        i += 1
        if cv2.waitKey(30) == 27 or i == 300:
            total = total/300
            print('total : %3f' % total)
            cv2.destroyAllWindows()
            break

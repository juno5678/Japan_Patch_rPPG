import numpy as np

import dlib
import cv2
from imutils import face_utils


class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.tracker = cv2.TrackerCSRT_create()

    def face_detect(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return

        rects = self.detector(frame, 1)
        #print(rects)
        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            #print(x,y,w,h)
            bbox = [x-int(w*0.1), y-int(h*0.1), int(w*1.2), int(h*1.1)]
            #face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            self.tracker.init(frame, bbox)
        else:
            print('face detect failed')
            return None
        return rects

    def face_track(self, frame):

        success, bbox = self.tracker.update(frame)
        if success:

            #face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            rect = dlib.rectangle(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
            rects = dlib.rectangles()
            rects.append(rect)
            #p1 = [bbox[0], bbox[1]]
            #p2 = [bbox[0]+bbox[2], bbox[1]+bbox[3]]
            #p2 = [int(point[0]) + int(1.25*w), int(point[1] + int(h*1.2))]
            #frame_out = frame.copy()
            #cv2.rectangle(frame_out, p1, p2, (0, 0, 255), 3)
            #cv2.imshow('face rectangle', frame_out)
            #cv2.imshow('face', face)
        else:
            print("tracking failed!!")
            return None
        return rects

    def detect_landmark(self, frame, face_rects):
        for k, d in enumerate(face_rects):
            shape = self.landmark_predictor(frame, d)
        return shape

    # get left eye, right eye and mouse's landmark
    def get_key_landmark(self, shape):
        return [shape.part(17), shape.part(26), shape.part(57)]



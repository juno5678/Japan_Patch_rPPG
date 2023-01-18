import numpy as np

import dlib
import cv2
from imutils import face_utils


class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        self.tracker1 = cv2.TrackerMIL_create()
        self.tracker2 = cv2.TrackerMIL_create()
        self.rgb_roi = [0, 0, 0, 0]
        self.rgb_bbox = [0, 0, 0, 0]
        self.rgb_diff_bbox = [0, 0, 0, 0]
        self.rgb_diff_roi_bbox = [0, 0, 0, 0]
        self.rgb_pre_bbox = [0, 0, 0, 0]
        self.rgb_not_found_count = 0
        self.rgb_first_detect = True
        self.nir_roi = [0, 0, 0, 0]
        self.nir_bbox = [0, 0, 0, 0]
        self.nir_diff_bbox = [0, 0, 0, 0]
        self.nir_diff_roi_bbox = [0, 0, 0, 0]
        self.nir_pre_bbox = [0, 0, 0, 0]
        self.nir_not_found_count = 0
        self.nir_first_detect = True

    def face_detect_rgb(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.rgb_first_detect:
            rects = self.detector(gray, 1)
            if len(rects) > 0:
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                self.rgb_roi = (max(0, x-round(1*w)), max(0, y-round(1*h)), round(w*3), round(h*3))
                self.rgb_first_detect = False
                print('first detect success!!')
            else:
                print('first detect failed!!')
            #cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if not self.rgb_first_detect:
            gray = gray[self.rgb_roi[1]:min(gray.shape[0], self.rgb_roi[1]+self.rgb_roi[3]),
                        self.rgb_roi[0]:min(gray.shape[1], self.rgb_roi[0]+self.rgb_roi[2])]

        print(self.rgb_roi)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        rects = self.detector(gray, 1)

        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            self.rgb_bbox = (self.rgb_roi[0]+x, self.rgb_roi[1]+y, w, h)
            self.rgb_diff_roi_bbox = (self.rgb_roi[0] - self.rgb_bbox[0], self.rgb_roi[1] - self.rgb_bbox[1],
                                      self.rgb_roi[2] - self.rgb_bbox[2], self.rgb_roi[3] - self.rgb_bbox[3])
            if self.rgb_pre_bbox[3] != 0:
                self.rgb_diff_bbox = (self.rgb_bbox[0] - self.rgb_pre_bbox[0], self.rgb_bbox[1] - self.rgb_pre_bbox[1],
                                      self.rgb_bbox[2] - self.rgb_pre_bbox[2], self.rgb_bbox[3] - self.rgb_pre_bbox[3])
            self.rgb_pre_bbox = self.rgb_bbox
            face_frame = frame[self.rgb_bbox[1]:self.rgb_bbox[1]+self.rgb_bbox[3],
                               self.rgb_bbox[0]:self.rgb_bbox[0]+self.rgb_bbox[2]]

            self.rgb_roi = (max(0, self.rgb_bbox[0]+self.rgb_diff_roi_bbox[0]+self.rgb_diff_bbox[0]),
                            max(0, self.rgb_bbox[1]+self.rgb_diff_roi_bbox[1]+self.rgb_diff_bbox[1]),
                            round(w*2), round(h*2))

        else:
            self.rgb_not_found_count += 1
            print("didn't found face %d times" % self.rgb_not_found_count)
            if self.rgb_bbox[0] > 0:
                face_frame = frame[self.rgb_bbox[1]:min(frame.shape[0], self.rgb_bbox[1]+self.rgb_bbox[3]),
                                   self.rgb_bbox[0]:min(frame.shape[1], self.rgb_bbox[0]+self.rgb_bbox[2])]
            else:
                print(self.rgb_bbox)
                print('first detect failed')
                return None, None

        return face_frame, self.rgb_bbox

    def face_detect_gray(self, frame):
        if frame is None:
            print("No frame to do face detection")
            return
        roi_img = frame.copy()
        if self.nir_first_detect:
            rects = self.detector(roi_img, 1)
            if len(rects) > 0:
                (x, y, w, h) = face_utils.rect_to_bb(rects[0])
                self.nir_roi = (max(0, x-round(0.5*w)), max(0, y-round(0.5*h)), round(w*2), round(h*2))
                self.nir_first_detect = False

        if not self.nir_first_detect:
            roi_img = frame[self.nir_roi[1]:min(frame.shape[0], self.nir_roi[1]+self.nir_roi[3]),
                            self.nir_roi[0]:min(frame.shape[1], self.nir_roi[0]+self.nir_roi[2])]

        rects = self.detector(roi_img, 1)

        if len(rects) > 0:
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            self.nir_bbox = (self.nir_roi[0]+x, self.nir_roi[1]+y, w, h)
            self.nir_diff_roi_bbox = (self.nir_roi[0] - self.nir_bbox[0], self.nir_roi[1] - self.nir_bbox[1],
                                      self.nir_roi[2] - self.nir_bbox[2], self.nir_roi[3] - self.nir_bbox[3])
            if self.nir_pre_bbox[3] != 0:
                self.nir_diff_bbox = (self.nir_bbox[0] - self.nir_pre_bbox[0], self.nir_bbox[1] - self.nir_pre_bbox[1],
                                      self.nir_bbox[2] - self.nir_pre_bbox[2], self.nir_bbox[3] - self.nir_pre_bbox[3])
            self.nir_pre_bbox = self.nir_bbox
            face_frame = frame[self.nir_bbox[1]:min(frame.shape[0], self.nir_bbox[1]+self.nir_bbox[3]),
                               self.nir_bbox[0]:min(frame.shape[1], self.nir_bbox[0]+self.nir_bbox[2])]
            self.nir_roi = (max(0, self.nir_bbox[0]+self.nir_diff_roi_bbox[0]+self.nir_diff_bbox[0]),
                            max(0, self.nir_bbox[1]+self.nir_diff_roi_bbox[1]+self.nir_diff_bbox[1]),
                            round(w*2), round(h*2))

        else:
            self.nir_not_found_count += 1
            print("didn't found face %d times" % self.nir_not_found_count)
            if self.nir_bbox[0] > 0:
                face_frame = frame[self.nir_bbox[1]:min(frame.shape[0], self.nir_bbox[1]+self.nir_bbox[3]),
                                   self.nir_bbox[0]:min(frame.shape[1], self.nir_bbox[0]+self.nir_bbox[2])]
            else:
                return None
        return face_frame, self.nir_bbox



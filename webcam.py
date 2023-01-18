import pyrealsense2 as rs
import numpy as np
import time
import cv2


class Camera_RGB(object):
    def __init__(self):
        self.camera_stream = 1
        self.cap = None
        t0 = 0

    def start(self):
        print("Start camera " + str(self.camera_stream))

        self.cap = cv2.VideoCapture(self.camera_stream)
        if not self.cap.isOpened():
            print("Cannot open canera")
            return
        self.t0 = time.time()
        self.valid = False
        try:
            ret, resp = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
            self.valid = True
        except:
            print("Can receive frame ")
            self.valid = False

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("Camera Stopped")

    def get_frame(self):
        if self.valid:
            _, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame is None:
                print("End of Camera")
                self.stop()
                print(time.time() - self.t0)
                return
            # else:
            # frame = cv2.resize(frame, (640, 480))
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Can not open canera)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame


class Webcam(object):
    def __init__(self):
        self.points = rs.points()
        self.pipeline = rs.pipeline()
        self.dirname = ""  # for nothing, just to make 2 inputs the same
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8,
                                  30)  # todo: avoid 20 fps, cause the app stop
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def start(self):
        print("Starting the webcam")
        profile = self.pipeline.start(self.config)
        time.sleep(0.5)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print("IR Emitter Status: " + str(depth_sensor.get_option(rs.option.emitter_enabled)))

    def get_frame(self):

        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames)

        aligned_color_frame = aligned_frames.get_color_frame()

        color_frame = frames.get_color_frame()
        nir_frame = frames.get_infrared_frame()

        aligned_color_out = np.asanyarray(aligned_color_frame.get_data())
        color_out = np.asanyarray(color_frame.get_data())
        nir_out = np.asanyarray(nir_frame.get_data(), dtype='uint8')
        aligned_nir = self.align_nir_to_rgb(nir_out, aligned_color_out)

        return color_out, aligned_nir

    def stop(self):
        self.pipeline.stop()
        print("Stopped the webcam")

    def align_nir_to_rgb(self, nir_frame, aligned_rgb):
        aligned_rect = self.find_align_nir_rect(aligned_rgb)
        [x, y, w, h] = aligned_rect
        aligned_nir = nir_frame[y:y + h, x:x + w]
        aligned_nir = cv2.resize(aligned_nir, (aligned_rgb.shape[1], aligned_rgb.shape[0]))
        return aligned_nir

    def find_align_nir_rect(self, aligned_rgb):
        gray_img = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
        th, binary = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        row = aligned_rgb.shape[0]
        col = aligned_rgb.shape[1]

        top = []
        bottom = []
        left = []
        right = []

        # find top
        for j in np.arange(col / 10, col, col / 10):
            for i in range(0, row):
                if binary[i, int(j)] != 0:
                    top.append(i)
                    # cv2.circle(rgb_output, (int(j), i), 5, (0, 0, 255), 3)
                    break

        # find bottom
        for j in np.arange(col / 10, col, col / 10):
            for i in range(row - 1, 0, -1):
                if binary[i, int(j)] != 0:
                    bottom.append(i)
                    # cv2.circle(rgb_output, (int(j), i), 5, (0, 0, 255), 3)
                    break

        # find left
        for i in np.arange(row / 10, row, row / 10):
            for j in range(0, col):
                if binary[int(i), j] != 0:
                    left.append(j)
                    # cv2.circle(rgb_output, (j, int(i)), 5, (0, 0, 255), 3)
                    break

        # find right
        for i in np.arange(row / 10, row, row / 10):
            for j in range(col - 1, 0, -1):
                if binary[int(i), j] != 0:
                    right.append(j)
                    # cv2.circle(rgb_output, (j, int(i)), 5, (0, 0, 255), 3)
                    break

        top_pt = min(top)
        bottom_pt = max(bottom)
        left_pt = min(left)
        right_pt = max(right)

        x = int(left_pt)
        y = int(top_pt)
        w = int(right_pt - left_pt)
        h = int(bottom_pt - top_pt)
        aligned_rect = [x, y, w, h]
        # [x, y, w, h] = cv2.boundingRect(contours[max_id])
        # print(x, y, w, h)
        # cv2.rectangle(rgb_output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.imshow('rgb output', rgb_output)
        # cv2.imshow('rgb binary', binary)
        return aligned_rect

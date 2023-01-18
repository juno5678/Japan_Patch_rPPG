import pyrealsense2 as rs
import numpy as np
import cv2


class Video(object):
    def __init__(self):
        self.points = rs.points()
        self.pipeline = rs.pipeline()
        self.dataPath = ""
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8,
                                  30)  # todo: avoid 20 fps, cause the app stop
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    def start(self):
        print("Starting the video" + self.dataPath)
        if self.dataPath == "":
            print("Invalid Filename!")
            return
        self.config.enable_device_from_file(self.dataPath, repeat_playback=False)
        profile = self.pipeline.start(self.config)
        profile.get_device().as_playback().set_real_time(False)

    def stop(self):
        self.pipeline.stop()
        print("Stopped the video")

    def get_frame(self):

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        nir_frame = frames.get_infrared_frame()
        color_out = np.asanyarray(color_frame.get_data())
        nir_out = np.asanyarray(nir_frame.get_data(), dtype='uint8')
        return color_out, nir_out


import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8,
                     30)  # todo: avoid 20 fps, cause the app stop
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

while 1:
    filt = rs.save_single_frameset()

    for x in range(5):
        pipeline.wait_for_frames()

    frame = pipeline.wait_for_frames()
    filt.process(frame)

    color_frame = frame.get_color_frame()
    color_out = np.asanyarray(color_frame.get_data())
    cv2.imshow('color', color_out)
    if cv2.waitKey(1) == 27:
        break
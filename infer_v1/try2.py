import pyrealsense2 as rs
import numpy as np
import cv2
from ourCamara import RealsenseCamera, Camera
cap = RealsenseCamera(True, 3)
# cap = Camera(1)
cap.start()
try:
    index = 1
    while True:
        frames = cap.read()
        # 获取对齐帧集
        frames = cap.align.process(frames)
        # 获取rgb图像
        color_rs = frames.get_color_frame()
        frame = np.asanyarray(color_rs.get_data())

        depth_rs = frames.get_depth_frame()

        depth_size = (depth_rs.get_height(), depth_rs.get_width())

        for filterMethod in cap.filter:
            depth_rs = filterMethod.process(depth_rs)

        depth_image = np.asanyarray(depth_rs.get_data())
        color_image = np.asanyarray(color_rs.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # print(type(depth_colormap))
        # <class 'numpy.ndarray'>

        # Stack both images horizontally（水平堆叠两个图像）
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)
finally:
    cap.stop()

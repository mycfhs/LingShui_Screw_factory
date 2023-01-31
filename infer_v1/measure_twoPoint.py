import numpy as np
import cv2
import ourFunc
import pyrealsense2 as rs
import time
from Camara import RealsenseCamera
cap = RealsenseCamera(False, 3)
cap.setCam()
# cap = Camera(1)
cap.start()

# try:
if True:
    index = 1
    while True:
        time_start = time.time()
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

        # 获得点云
        cap.pc.map_to(color_rs)
        points_rs = cap.pc.calculate(depth_rs)
        points = np.asanyarray(points_rs.get_vertices())
        dists = points.reshape(depth_size)
        # (x1, y1), (x2, y2) = ourFunc.click_to_get_points(np.asanyarray(depth_rs.get_data()))
        (x1, y1), (x2, y2) = ourFunc.click_to_get_points(frame)
        # (x1, y1), (x2, y2) = ourFunc.click_to_get_points(dimg)

        # print(dists.shape)  # 480, 640      (y,x)
        point1 = dists[y1][x1]
        point2 = dists[y2][x2]

        dis = [point1[i]-point2[i] for i in range(3)]
        print(point1, point2)
        print('distance:', ourFunc.calc_dist(dis))
        cv2.waitKey(0)
        break

# finally:
#     pipeline.stop()


"""
    Author:mycf
"""

import pyrealsense2 as rs
import threading
import time
from ourFunc import print_color
import cv2
import numpy as np


class RealsenseCamera:
    def __init__(self, setDefault=False, setDefaultTime=50):
        self.frames = None
        self.stopped = False
        self.setDefault = setDefault
        self.setDefaultTime = setDefaultTime

        self.color_rs = None
        self.depth_rs = None

        self.pipeline = None  # 摄像头
        self.profile = None  # 用于调摄像头参数
        self.align = None  # 用来对其深度和rgb图像

        # 滤波器
        self.use_decimation = False
        self.use_spatial = True
        self.use_threshold_filter = True
        self.use_temporal = False
        self.use_hole_filling = False

        self.decimation = None
        self.spatial = None
        self.threshold_filter = None
        self.temporal = None
        self.hole_filling = None

        self.filter = []
        self.init()

    def init(self):
        # 创建 config 对象：
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline = rs.pipeline()
        if self.setDefault:
            self.setDefaultOfCam(self.setDefaultTime)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

        self.setCam()
        self.setFilter()

    def start(self):
        """
        由于计算耗时较多，性能会降低
        故采用多线程摄像头
        :return:
        """
        threading.Thread(target=self.update, args=()).start()
        print_color('wait 1 second for cam thread to read img.')
        time.sleep(3)
        print_color('Done!')

    def update(self):
        while True:
            if self.stopped:
                return
            self.frames = self.pipeline.wait_for_frames()

    def read(self):
        # 获取对齐帧集
        frames = self.align.process(self.frames)
        # 获取rgb图像
        color_rs = frames.get_color_frame()
        frame = np.asanyarray(color_rs.get_data())

        depth_rs = frames.get_depth_frame()

        depth_size = (depth_rs.get_height(), depth_rs.get_width())

        for filterMethod in self.filter:
            depth_rs = filterMethod.process(depth_rs)

        # 获得点云
        self.pc.map_to(color_rs)
        points_rs = self.pc.calculate(depth_rs)
        points = np.asanyarray(points_rs.get_vertices())

        dist = points.reshape(depth_size)
        return frame, dist

    def stop(self):
        self.stopped = True

    def setDefaultOfCam(self, waitTime=50) -> None:
        """
        重置相机，防止其他队伍修改的参数影响自己
        以防万一，时间可以设置的稍微长一些
        :param waitTime: 重置相机后立刻读取会出问题
        :return:
        """
        profile = self.pipeline.start()
        profile.get_device().hardware_reset()
        self.pipeline.stop()

        print_color("Initializing the camera...")
        for i in range(1, waitTime + 1):
            print(waitTime + 1 - i)
            time.sleep(1)
        print_color("Done!")
        return None

    def setParamCam(self, sensor, param: str, value: float = None) -> None:
        """
        设置相机的参数
        :param sensor:
        :param param: 变量名  字符串格式
        :param value:
        :return:
        """
        print_color(f"Setting param: {param}")
        try:
            paramRange = eval('sensor.get_option_range(rs.option.%s)' % param)
            try:
                rawValue = paramRange.default
                if value is None:
                    value = rawValue
                if eval('sensor.supports(rs.option.%s)' % param):
                    eval('sensor.set_option(rs.option.%s, value)' % param)
                    print_color(f"param '{param}' is supported! From {rawValue} changed to {value}. "
                                f"({paramRange}")
                    description = eval('sensor.get_option_description(rs.option.%s)' % param)
                    print(f"description of '{param}':\n{description}")
                else:
                    print_color(f"ERROR! param '{param}' is NOT supported! Ignore.",
                                "red", "normal", "black")

            except:
                print_color(f"ERROR! {value} not in {paramRange}. Using default:{paramRange.default}",
                            "red", "normal", "black")

        except:
            print_color(f"ERROR! '{param}' is invalid!",
                        "red", "normal", "black")
        return None

    def setCam(self) -> None:
        """
        设置相机参数
        各参数意义见：https://blog.csdn.net/FlowMytears/article/details/125963526
        :return:
        """
        print_color('start setting params of cam')

        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        # print(depth_sensor.get_supported_options()) # 查看可支持选项。 以下为sr305(sr300)支持的。 有注释的重点调参修改即可。
        self.setParamCam(depth_sensor, 'visual_preset')
        self.setParamCam(depth_sensor, 'laser_power')
        self.setParamCam(depth_sensor, 'accuracy', 1)
        self.setParamCam(depth_sensor, 'motion_range', 180.0)  # 静态调大 动态调小
        self.setParamCam(depth_sensor, 'filter_option', 6)  # 滤波器选择
        self.setParamCam(depth_sensor, 'confidence_threshold', 3)  # 每个点的深度信息置信度，调低点吧，总比‘0’强
        self.setParamCam(depth_sensor, 'frames_queue_size', 16.0)  # 在给定的时间内你可以保持的最大帧数。增加这个数字将减少帧下降，但增加延迟，反之亦然

        rgb_sensor = device.first_color_sensor()
        # print(rgb_sensor.get_supported_options())
        self.setParamCam(rgb_sensor, 'backlight_compensation')  # 背光补偿
        self.setParamCam(rgb_sensor, 'brightness')  # 短波紫外线图像亮度
        self.setParamCam(rgb_sensor, 'contrast')  # 短波紫外线图像对比
        self.setParamCam(rgb_sensor, 'exposure')  # 控制彩色相机的曝光时间。设置任何值都将禁用自动曝光
        self.setParamCam(rgb_sensor, 'gain')  # 短波紫外线图像增益
        self.setParamCam(rgb_sensor, 'gamma')
        self.setParamCam(rgb_sensor, 'hue')  # 图像饱和度设置
        self.setParamCam(rgb_sensor, 'saturation')
        self.setParamCam(rgb_sensor, 'sharpness')
        self.setParamCam(rgb_sensor, 'white_balance')
        self.setParamCam(rgb_sensor, 'enable_auto_exposure')  # 启用/禁用自动曝光
        self.setParamCam(rgb_sensor, 'enable_auto_white_balance')
        self.setParamCam(rgb_sensor, 'frames_queue_size')
        return None

    def setParamFilter(self, filterName, param: str, value: float = None) -> None:
        """
        设置滤波器们的参数
        :param filterName: 滤波器名字
        :param param: 参数名称
        :param value: 参数值
        :return:
        """
        print_color(f"Setting param: {param}")
        try:
            paramRange = eval('self.%s.get_option_range(rs.option.%s)' % (filterName, param))
            try:
                rawValue = paramRange.default
                if value is None:
                    value = rawValue

                eval('self.%s.set_option(rs.option.%s, value)' % (filterName, param))
                print_color(f"param '{param}' is supported! From {rawValue} changed to {value}. "
                            f"({paramRange}")
                description = eval('self.%s.get_option_description(rs.option.%s)' % (filterName, param))
                print(f"description of '{param}':\n{description}")

            except:
                print_color(f"ERROR! {value} not in {paramRange}. Using default:{paramRange.default}",
                            "red", "normal", "black")

        except:
            print_color(f"ERROR! '{param}' is invalid!",
                        "red", "normal", "black")
        return None

    def setFilter(self):
        """
        设置滤波器 处理图像
        :return:
        """
        if self.use_decimation:
            self.decimation = rs.ddecimation_filter()
            self.setParamFilter('decimation', 'filter_magnitude')

        if self.use_spatial:
            self.spatial = rs.spatial_filter()
            self.setParamFilter('spatial', 'filter_magnitude', 1)
            self.setParamFilter('spatial', 'filter_smooth_alpha')
            self.setParamFilter('spatial', 'filter_smooth_delta')
            self.setParamFilter('spatial', 'holes_fill', 4)

        if self.use_threshold_filter:
            self.threshold_filter = rs.threshold_filter()
            self.setParamFilter('threshold_filter', 'min_distance', 0.7)
            self.setParamFilter('threshold_filter', 'max_distance', 1.4)

        if self.use_hole_filling:
            self.hole_filling = rs.hole_filling_filter()
            self.setParamFilter('hole_filling', 'holes_fill')

        if self.use_temporal:
            self.temporal = rs.temporal_filter()
            self.setParamFilter('temporal', 'filter_smooth_alpha')
            self.setParamFilter('temporal', 'filter_smooth_delta')
            self.setParamFilter('temporal', 'holes_fill')

        for i in ['decimation', 'spatial', 'threshold_filter', 'hole_filling', 'temporal']:
            if eval('self.use_%s' % i):
                self.filter.append(eval('self.%s' % i))


class Camera:
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.fps = -1
        for _ in range(10):  # warm up the camera
            (self.grabbed, self.frame) = self.stream.read()

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        print_color('wait 1 second for cam thread to read img.')
        time.sleep(1)
        print_color('Done!')

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def testCam(src=0):
    import cv2 as cv
    # 打开摄像头
    cap = cv.VideoCapture(src)

    while True:
        # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
        hx, frame = cap.read()

        # 如果hx为Flase表示开启摄像头失败，那么就输出"read video error"并退出程序
        if hx is False:
            # 打印报错
            print('read video error')
            # 退出程序
            exit(0)

        # 显示摄像头图像，其中的video为窗口名称，frame为图像
        cv.imshow('video', frame)

        # 监测键盘输入是否为q，为q则退出程序
        if cv.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break

    # 释放摄像头
    cap.release()

    # 结束所有窗口
    cv.destroyAllWindows()

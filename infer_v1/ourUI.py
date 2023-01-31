from PyQt5 import QtCore, QtGui, QtWidgets
# import message
# import socket
# import socket_round1st
import threading
import ourCamara
from ourSegModel import load_model, display_and_measure, FastBaseTransform
from ourMeasureFun import *
import torch
import time


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = ourCamara.RealsenseCamera(True, 3)  # 视频流

        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.model_init()   # 初始化模型
        self.detection_start_statu = False

        self.predict_model = None
        self.processed_frame = None

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_start = QtWidgets.QPushButton('开始检测')
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_start.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.button_start.move(10, 50)
        self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(641, 481)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_start)
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.button_start.clicked.connect(self.detection_start)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            try:
                self.cap.start()
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
            except:
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)

            # threading.Thread(target=self.update, args=()).start()
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.stop()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    '''自己的函数'''
    def model_init(self):
        print_color('Loading Model...')
        self.net = load_model()
        print_color('Model loaded! Ready to measure!')
        pass

    def show_camera(self):
        if self.detection_start_statu and self.processed_frame is not None:
            show = self.processed_frame
        else:
            show, _ = self.cap.read() # 从视频流中读取
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def detection_start(self):
        # Round_num = 3
        # phone = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # phone.connect(('192.168.1.66', 6666))
        # socket_round1st.send_start(Round_num, phone)

        if not self.timer_camera.isActive():
            msg = QtWidgets.QMessageBox.warning(self, 'warning', "未开启摄像头", buttons=QtWidgets.QMessageBox.Ok)
        else:
            if not self.detection_start_statu:
                self.detection_start_statu = True
                self.button_start.setText('结束检测')
                threading.Thread(target=self.detection_process, args=()).start()
            else:
                self.detection_start_statu = False
                self.button_start.setText('开始检测')

    def detection_process(self):
        """
        由于计算耗时较多，性能会降低
        故采用多线程来计算
        :return:
        """
        with torch.no_grad():
            while self.detection_start_statu:
                frame, dist = self.cap.read()
                frame = torch.from_numpy(frame).cuda().float()

                time_start = time.time()
                preds = self.net(FastBaseTransform()(frame.unsqueeze(0)),
                                 extras={"backbone": "full", "interrupt": False,
                                         "keep_statistics": False,
                                         "moving_statistics": None})["pred_outs"]

                self.processed_frame, masks, classes, scores, boxes = display_and_measure(preds, frame, time_start)
                for i in range(len(classes)):
                    if scores[i]<0.8:
                        break
                    x1, y1, x2, y2 = boxes[i]
                    w, h = x2-x1, y2-y1
                    centerPoint = [(x1+x2)//2, (y1+y2)//2]
                    if classes[i] == 0:
                        pass
                        # ents = mask2point(masks[i], [y1, x1, y2, x2])
                        # print(cv2.fitLine(np.array(ents), cv2.DIST_L2, 0, 0.01, 0.01))
                        # print(measureBolt(ents, img, dist))
                        # break
                    elif classes[i] == 1:
                        pass
                        # ents = mask2point(masks[i], [y1, x1, y2, x2])
                        # ents = pureGasketEnts(ents, centerPoint)
                        # print(boxes[i])
                        #  = boxes[i]
                        # try:
                        # if True:
                        #     outLen, inLen, frame = measure_gasket(ents, frame, dist, centerPoint, w, h)
                        #     print(outLen, inLen)
                        #     if inLen==-2:
                        #         continue
                        # except:
                        #     continue



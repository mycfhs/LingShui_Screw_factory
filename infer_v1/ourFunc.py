"""
    Functions of our project.
"""
import cv2
import numpy as np
import os
from PIL import Image


def print_color(content, text_color: str = "blue", style: str = "bold", background: str = "black") -> None:
    escseq = "\033["
    style = str(style).lower()
    text_color = str(text_color).lower()
    background = str(background).lower()
    if text_color == "black":
        text_color = "30;"
    elif text_color == "red":
        text_color = "31;"
    elif text_color == "green":
        text_color = "32;"
    elif text_color == "yellow":
        text_color = "33;"
    elif text_color == "blue":
        text_color = "34;"
    elif text_color == "purple":
        text_color = "35;"
    elif text_color == "cyan":
        text_color = "36;"
    elif text_color == "white":
        text_color = "37;"
    if style == "normal":
        style = "0;"
    elif style == "bold":
        style = "1;"
    elif style == "light":
        style = "2;"
    elif style == "italicized":
        style = "3;"
    elif style == "underline":
        style = "4;"
    elif style == "blink":
        style = "5;"
    if background == "black":
        background = "40"
    elif background == "red":
        background = "41"
    elif background == "green":
        background = "42"
    elif background == "yellow":
        background = "43"
    elif background == "blue":
        background = "44"
    elif background == "purple":
        background = "45"
    elif background == "cyan":
        background = "46"
    elif background == "white":
        background = "47"
    stopper = "m"
    print(escseq + style + text_color + background + stopper + str(content) + escseq + stopper)


def click_to_get_points(img):
    """
    通过点击图片上的点来获得坐标
    由于个人需求 仅返回前两个点
    :param img:图片
    :return:前两个点的坐标
    """

    a, b = [], []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    return (a[0], b[0]), (a[1], b[1])


def calc_dist(xyz: list) -> float:
    """
    计算n维向量的模
    :param xyz:
    :return:
    """
    result = 0
    for i in xyz:
        result += i ** 2
    return result ** 0.5


def mask2point(ent, bbox):
    newEnt = []
    for i in range(bbox[0] - 2, bbox[2] + 2):
        li = []
        for j in range(bbox[1] - 2, bbox[3] + 2):
            if ent[i][j] != ent[i][j + 1]:
                li.append((j, i))
        if len(li) != 0:
            newEnt.append(li[0])
            newEnt.append(li[-1])
    return newEnt


def img_paste(image, polygon_list):
    """
    切割出多边形的图像
    :param image: 原图
    :param polygon_list: 多边形的所有点
    :return: 可以直接被cv2读取的图像
    """
    im = np.zeros(image.shape[:2], dtype="uint8")
    b = np.array(polygon_list, dtype=np.int32)
    roi_t = [b[j] for j in range(len(polygon_list))]

    roi_t = np.expand_dims(np.asarray(roi_t), axis=0)
    cv2.polylines(im, roi_t, True, (0, 255, 0))
    cv2.fillPoly(im, roi_t, 255)

    masked = cv2.bitwise_and(image, image, mask=im)
    b, g, r = cv2.split(masked)
    return cv2.merge([r, g, b])


def Lineleastsq(ents):
    from scipy.optimize import leastsq
    x, y = [], []
    for ent in ents:
        x.append(ent[0])
        y.append(ent[1])
    Xi = np.array(x)
    Yi = np.array(y)

    def error(p, x, y):
        def func(p, x):
            k, b = p
            return k * x + b
        return func(p, x) - y

    k, b = leastsq(error, [1, 1], args=(Xi, Yi))[0]
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    plt.scatter(Xi, Yi, color="green", label="样本数据", linewidth=2)
    x = np.linspace(240,320,100)
    y = k * x + b
    plt.plot(x, y, color="red", label="拟合直线", linewidth=2)
    plt.title('y={}+{}x'.format(b, k))
    plt.legend(loc='lower right')
    plt.show()
    return k, b


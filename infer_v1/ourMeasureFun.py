import cv2
from ourFunc import *


def pureGasketEnts(ents, center):
    try:
        dist = []
        newEnts = []
        flag = False  # 是否有删除点
        for i in ents:
            dist.append(calc_dist([i[0] - center[0], i[1] - center[1]]))
        aveDist = sum(dist) // len(dist)
        for i in ents:
            if calc_dist([i[0] - center[0], i[1] - center[1]]) > aveDist * 0.4:
                newEnts.append(i)
            else:
                flag = True
        return newEnts, flag
    except:
        return ents, False


def measure_gasket(ents, frame, dists, centerPoint, w, h):
    # 先计算第一个
    # getOutGasket = False
    #
    # while not getOutGasket:
    (ex1, ey1), (ew1, eh1), anle1 = cv2.fitEllipse(np.array(ents))
    #     # print(ex1 - centerPoint[0],ey1 - centerPoint[1],  ew1 - w , eh1 - h, anle1)
    #     if -2 < ex1 - centerPoint[0] < 2 and -2 < ey1 - centerPoint[1] < 2 \
    #             and -10 < ew1 - w < 10 and -5 < eh1 - h < 5:
    #         getOutGasket = True
    #     else:
    #         ents, flag = pureGasketEnts(ents, centerPoint)
    #         if flag and len(ents) > 10:
    #             continue
    #         else:
    #             break
    #             # print('This gasket ignore!')
    #             # return -2, -2, frame

    ex1, ey1, ew1, eh1, anle1 = map(int, [ex1, ey1, ew1 / 2, eh1 / 2, anle1])
    cv2.ellipse(frame, (ex1, ey1), (ew1, eh1), anle1, 0, 360, (255, 0, 0), 2)
    outLen, inLen = -1, -1
    outLenList = []
    inLenList = []

    l1 = cv2.ellipse2Poly((ex1, ey1), (ew1, eh1), anle1, 0, 180, 5)
    l2 = cv2.ellipse2Poly((ex1, ey1), (ew1, eh1), anle1, 180, 360, 5)
    for i in range(len(l1)):
        tx1, ty1 = l1[i]
        tx2, ty2 = l2[i]
        point1 = dists[ty1][tx1]
        point2 = dists[ty2][tx2]
        if sum(point1) > 0 and sum(point2) > 0:
            outLenList.append(calc_dist([point1[i] - point2[i] for i in range(3)]))

    outLenList.sort()
    outListlength = len(outLenList)
    outLen = sum(outLenList[outListlength // 3:outListlength // 3 * 2]) / outListlength * 3

    # 寻找螺栓的内环
    masked_image = img_paste(frame, ents)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contoursEnts in list(contours):
        # 会找到很多轮廓 要筛选 先筛去点很少的 然后拟合椭圆并筛去不合适的
        if len(contoursEnts) > 10:
            ents = contoursEnts
            getInGasket = False
            while not getInGasket:
                (ex2, ey2), (ew2, eh2), anle2 = cv2.fitEllipse(np.array(ents))
                if -2 < ex2 - centerPoint[0] < 2 and -2 < ey2 - centerPoint[1] < 2:
                    getInGasket = True
                else:
                    ents, flag = pureGasketEnts(ents, centerPoint)
                    if flag and len(ents) > 10:
                        continue
                    else:
                        break

            ex2, ey2, ew2, eh2, anle2 = map(int, [ex2, ey2, ew2 / 2, eh2 / 2, anle2])

            # 如果轮廓和外椭圆太远、比外椭圆还大、长短轴之比严重不符则不要
            if calc_dist([ex2 - ex1, ey2 - ey1]) < 3 and ew2 > 0 and ew1 > 0 \
                    and ew2 < ew1 * 0.8 and eh2 < eh1 * 0.8 and abs(ew2 / eh2 - ew1 / eh1) < 0.1:

                cv2.ellipse(frame, (ex2, ey2), (ew2, eh2), anle2, 0, 360, (255, 0, 0), 2)
                l1 = cv2.ellipse2Poly((ex2, ey2), (ew2, eh2), anle2, 0, 180, 5)
                l2 = cv2.ellipse2Poly((ex2, ey2), (ew2, eh2), anle2, 180, 360, 5)
                for i in range(len(l1)):
                    tx1, ty1 = l1[i]
                    tx2, ty2 = l2[i]
                    point1 = dists[ty1][tx1]
                    point2 = dists[ty2][tx2]
                    if sum(point1) > 0 and sum(point2) > 0:
                        inLenList.append(calc_dist([point1[i] - point2[i] for i in range(3)]))

                inLenList.sort()
                inListlength = len(inLenList)
                if inListlength > 0:
                    inLen = sum(inLenList[inListlength // 3:inListlength // 3 * 2]) / inListlength * 3
                    break
        if inLen == -1:
            inLen = 0.6 * outLen
    return outLen, inLen, frame


def measureBolt(ents, frame, dists):
    k, b = Lineleastsq(ents)
    pointDista = []
    pointDistb = []
    for ent in ents:
        x0, y0 = ent
        dis = (k * x0 - y0 + b) / ((1 + k ** 2) ** 0.5)
        if dis < 0:
            pointDista.append([dis, x0, y0])
        elif dis > 0:
            pointDistb.append([dis, x0, y0])
    pointDista = sorted(pointDista, key=lambda i: i[0])
    lengtha = len(pointDista)
    pointDista = pointDista[lengtha // 3:lengtha // 3 * 2]

    pointDistb = sorted(pointDistb, key=lambda i: i[0])
    lengthb = len(pointDistb)
    pointDistb = pointDistb[lengthb // 3:lengthb // 3 * 2]

    for i in range(lengtha // 3):
        _, tx1, ty1 = pointDista[i]
        _, tx2, ty2 = pointDistb[i]
        point1 = dists[ty1][tx1]
        point2 = dists[ty2][tx2]
    print(pointDista)
    print(pointDistb)
    return None

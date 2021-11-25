'''
检查大图片中是否包含小图片
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_match_by_FLANN(target_address, template_address, min_match_count=10):
    '''
    #基于FLANN的匹配器(FLANN based Matcher)定位图片
    :param target_address:被检查图片路径
    :param template_address:模板图片路径
    :param min_match_count:最低特征点匹配数量
    :return:模板图片所在位置中心坐标
    '''
    x = 0
    y = 0
    matches_mask = None
    min_match = min_match_count  # 设置最低特征点匹配数量
    template = cv2.imread(template_address, 0)  # queryImage
    target = cv2.imread(target_address, 0)  # trainImage
    # Initiate SIFT detector创建sift检测器
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > min_match:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        for val in dst:
            print(val)
            x += val[0][0]
            y += val[0][1]
        x = x / 4
        y = y / 4
        print('The center position of matching img is [%f,%f]' % (x, y))
    else:
        print("Not enough matches are found - %d/%d" % (len(good), min_match))
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    plt.imshow(result, 'gray')
    plt.show()

    return x, y

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]
    # print("order:",order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

def template(img_gray, template_img, template_threshold):
    '''
    img_gray:待检测的灰度图片格式
    template_img:模板小图，也是灰度化了
    template_threshold:模板匹配的置信度
    '''

    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    #start_time = time.time()
    loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标
    score = res[res >= template_threshold]  # 大于模板阈值的目标置信度
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接
    thresh = 0.3  # NMS里面的IOU交互比阈值

    keep_dets = py_nms(data_hstack, thresh)
    #print("nms time:", time.time() - start_time)  # 打印数据处理到nms运行时间
    dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
    return dets

def img_match_by_RGB2gray(target_address,template_address,template_shreshold = 0.8):
    '''
    识别是否包含模板图片
    :param template_address: 模版图地址
    :param template_shreshold: 模版识别率
    :return: 如果包含模板图片，则返回模板图片中心坐标;否则返回(0,0)
    '''

    # 以灰度加载图片
    img_rgb = cv2.imread(target_address)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # 模板相似度
    temp_shreshold = template_shreshold

    img_template = cv2.imread(template_address, 0)

    x = 0
    y = 0
    try:
        dets = template(img_gray, img_template, temp_shreshold)
        for coord in dets:
            np.any(dets >= 0)
            x = (int(coord[0]) + (int(coord[2]))) / 2
            y = (int(coord[1]) + (int(coord[3]))) / 2
            print("在", (x,y), "检测到",template_address)
            #识别区画红框
            #cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
            break
        #打开测试窗口
        #cv2.imshow('img_rgb', img_rgb)
        #cv2.waitKey(0)
    except:
        pass
    return x,y


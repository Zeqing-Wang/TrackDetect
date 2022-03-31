# -*- coding: utf-8 -*-
# @Time    : 2022/2/13 12:27
# @Author  : naptmn
# @File    : get_features.py
# @Software: PyCharm
import numpy as np
import os
import torch
def get_feature(begin = 0, end = 0):
    path = '../../yolov5-6.0/features/'
    pathlist = os.listdir(path)
    # print(os.listdir(path))
    features = None
    for i in pathlist:
        feature = np.load(path+i)
        feature = np.squeeze(feature)
        if features is None:
            features = feature
        else:
            features = np.vstack((features, feature))
        # print(feature.shape)
        # features.append(feature)
    return torch.tensor(features)

def read_file(path):
    feature = np.load(path)
    feature = np.squeeze(feature)
    return feature
def get_feature_thread():
    path = '../../yolov5-6.0/features/'
    pathlist = os.listdir(path)
    # print(os.listdir(path))
    features = None
    # 线程池，取结果时会阻塞
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(8) as executor:  # 创建 ThreadPoolExecutor
        future_list = [executor.submit(read_file, path+file) for file in pathlist]  # 提交任务
    for future in as_completed(future_list):
        if features is None:
            features = future.result()  # 获取任务结果
        else:
            features = np.vstack((features, future.result()))
    return torch.tensor(features)

    #print(time.time() - start_time)


if __name__ == '__main__':
    # 测试
    print(np.array(get_feature()).shape)
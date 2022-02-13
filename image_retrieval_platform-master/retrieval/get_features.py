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

if __name__ == '__main__':
    # 测试
    print(np.array(get_feature()).shape)
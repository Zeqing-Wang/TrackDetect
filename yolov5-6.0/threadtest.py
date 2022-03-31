# -*- coding: utf-8 -*-
# @Time    : 2022/2/17 1:54
# @Author  : naptmn
# @File    : threadtest.py
# @Software: PyCharm
import _thread
import detect
if __name__ =='__main__':
    gpulist = ['0', '1', '2', '3']
    threadnum = 0
    ok = True
    while ok:
        for gpu in gpulist:
            try:
                _thread.start_new_thread(detect.threadfunc(gpu))
                threadnum += 1
            except:
                print("线程达到上限，线程数为：",threadnum)
                ok = False
                break

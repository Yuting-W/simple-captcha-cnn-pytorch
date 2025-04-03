import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# 保存输入数据
def px(prefix, img):
    with open('./data/' + prefix + '_images.py', 'a+') as f:
        for x in range(4):
            print("[", file=f)
            for line in img[x]:
                l = line.astype(int).tolist()
                print(l, file=f, end=",\n")
            print("],\n", file=f)
        f.flush()

# 保存标签数据
def py(prefix, code):
    with open('./data/' + prefix + '_labels.py', 'a+') as f:
        for x in range(4):
            tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp[int(code[x])] = 1
            print(tmp, file=f, end=",\n")
        f.flush()
# 预处理图片
def paa(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    # img = img.crop((2, 1, 66, 22)) # 裁掉边变成 64x21
    img = img[1:-2, 2:-2]

    ret, img = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)  # 二值化
    # 分离数字
    img1 = img[:, 0:16]
    img2 = img[:, 16:32]
    img3 = img[:, 32:48]
    img4 = img[:, 48:64]

    return [img1, img2, img3, img4]

def work(prefix):
    results = {}
    files = []
    with open('./' + prefix + '.txt') as f:
        for x in f.readlines():
            tmp = x.split(':')
            files.append(tmp[0])
            results[tmp[0]] = tmp[1]

    with open('./data/' + prefix + '_images.py', 'w') as f:
        print("data = [", file=f)
    with open('./data/' + prefix + '_labels.py', 'w') as f:
        print("data = [", file=f)

    for file in files:
        print(file)
        img = paa(file)
        px(prefix, img)
        py(prefix, results[file])

    with open('./data/' + prefix + '_images.py', 'a+') as f:
        print("]", file=f)
    with open('./data/' + prefix + '_labels.py', 'a+') as f:
        print("]", file=f)

work('test')
work('train')

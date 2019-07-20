# 简化 ： 假设其中一个轴为0


def getRange(pts, ax):
    mx = pts[0][ax]
    mi = pts[0][ax]
    for p in pts:
        if p[ax] < mi:
            mi = p[ax]
        if p[ax] > mx:
            mx = p[ax]
    return mx - mi


def pts2flatten(pts):
    ret = []
    rg = [getRange(pts, i) for i in range(3)]
    deli = rg.index(min(rg))
    print("deli:", deli)
    rsvi = [i for i in range(3) if i != deli]
    for p in pts:
        ret.append([p[rsvi[0]], p[rsvi[1]]])
    return np.array(ret)


import math
import numpy as np
from PIL import Image

size = 100


def pts2image(pts):
    pts = np.array(pts)
    x_range = [min(pts[:, 0]), max(pts[:, 0])]
    y_range = [min(pts[:, 1]), max(pts[:, 1])]
    x_w = int((x_range[1] - x_range[0]) / size)
    y_w = int((y_range[1] - y_range[0]) / size)

    arr = np.zeros((y_w + 1, x_w + 1), dtype=np.uint)
    print(arr.shape)
    # print(len(pts))
    for p in pts:
        x = int((p[0] - x_range[0]) / size)
        y = int((p[1] - y_range[0]) / size)
        # print(x,y)
        arr[y][x] = 255
    # for i in range(1,min(len(arr),len(arr[0]))):
    #    arr[i][i] = 177
    print(arr.shape)
    img = Image.fromarray(arr, '1')
    img.save('tmp.png')


def corse(arr, x, y, val):
    if val <= 0:
        return arr
    w,h = arr.shape
    if 0 <= x < w and 0<=y<h:
        arr[x][y] = min(255, arr[x][y] + val)
    corse(arr, x - 1, y, val - 50)
    corse(arr, x, y - 1, val - 50)
    corse(arr, x + 1, y, val - 50)
    corse(arr, x, y + 1, val - 50)


def pos2mnistlike(x, y):
    size = 28
    arr = np.zeros((size, size))
    x_range = [min(x), max(x)]
    y_range = [min(y), max(y)]
    x_w = (x_range[1] - x_range[0]) / (size - 1)
    y_w = (y_range[1] - y_range[0]) / (size - 1)

    for i in range(len(x)):
        _x = int((x[i] - x_range[0]) / x_w)
        _y = int((y[i] - y_range[0]) / y_w)
        # arr[_x][_y] = min(arr[_x][_y] + 50, 255)
        corse(arr, _x, _y, 100)
    return arr


# import pytesseract
from PIL import Image


#


#
def getNumAns(pts):
    image = Image.open('tmp.jpg')
    code = pytesseract.image_to_string(image)
    return code


import numpy as np

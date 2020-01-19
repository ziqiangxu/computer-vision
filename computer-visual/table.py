import cv2 as cv
import skimage
import numpy as np
from skimage import morphology


def show_img(name: str, img: np.ndarray):
    cv.namedWindow(name, 0)
    cv.resizeWindow(name, 800, 1600)
    cv.imshow(name, img)


img_path = "../data/img/lenna.jpg"
img = cv.imread(img_path)
show_img("raw", img)

img_gray: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, img_thresh = cv.threshold(img_gray, 120, 255, cv.THRESH_BINARY_INV)
img_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                  cv.THRESH_BINARY_INV, 111, 10)
show_img("thresh", img_thresh)

# 先腐蚀，后膨胀 ---> 开运算， 去噪声
kernel = morphology.rectangle(3, 3)
img_close = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel)
img_open = cv.morphologyEx(img_close, cv.MORPH_OPEN, kernel)
show_img("open", img_open)

h, w = img_gray.shape
print(w, h)

# 先腐蚀，后膨胀 ---> 开运算
# 得到线图
kernel = np.ones((1, w // 30), np.uint8)  # 一条横线
show_img('kernel', kernel)
lines_horizontal = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)
show_img("horizontal", lines_horizontal)

kernel = np.ones((h // 30, 1), np.uint8)  # 一条竖线
lines_vertical = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)
show_img("vertical", lines_vertical)

# 得到网格图
img_grid = lines_vertical + lines_horizontal
show_img("grid", img_grid)

# 方案一： 依据连通域定位原始图像

# 方案二： 找到网格图的轮廓 【直线拟合？】
contours, hierarchy = cv.findContours(img_grid, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours)
print(len(contours))
img_contours = cv.drawContours(img, contours, -1, (0, 0, 200), 5)
show_img("contours", img_contours)

# 方案三： 先按行分割，再按列分割

# img_distance = cv.distanceTransform(img_grid, cv.)

# 得到横纵直线的交点
# img_dots = cv.bitwise_and(lines_vertical, lines_horizontal)
# show_img("dots", img_dots)

cv.waitKey(0)


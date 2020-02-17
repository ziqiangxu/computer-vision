import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_gbr = cv.imread("../data/img/lenna.jpg", 1)
img = cv.cvtColor(img_gbr, cv.COLOR_BGR2RGB)
plt.subplot(231)
plt.imshow(img)


def image_crop():
    img_crop = img[200:300, 200:300, :]
    return img_crop


# cooler image(by increase the value of blue channel)
def color_shift():
    shift_size = 40
    b_lim = 255 - shift_size
    r, g, b = cv.split(img)
    # split函数返回的通道顺序就是传入图像的通道顺序（ndarray对象并不带有顺序信息）
    # b1, g1, r1 = cv.split(img_gbr)
    b[b > b_lim] = 255
    b[b <= b_lim] = b[b <= b_lim] + shift_size
    img_shift = cv.merge([r, g, b])
    return img_shift


def rotation():
    mat = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), 70, 1)
    return cv.warpAffine(img, mat, dsize=(img.shape[0], img.shape[1]))


img1_gbr = cv.imread("./timg.jpeg")
img1 = cv.cvtColor(img1_gbr, cv.COLOR_BGR2RGB)


def perspective_transform():
    row_n, col_n, _ = img1.shape
    pts_1 = np.float32([[241,294], [306, 294], [206, 347], [310, 347]])
    pts_2 = np.float32([[206,294], [306, 294], [206, 347], [310, 347]])
    mat = cv.getPerspectiveTransform(pts_1, pts_2)
    return cv.warpPerspective(img1, mat, (col_n, row_n))


plt.subplot(232)
plt.imshow(image_crop())
plt.subplot(233)
plt.imshow(color_shift())
plt.subplot(234)
plt.imshow(rotation())
plt.subplot(235)
plt.imshow(img1)
plt.subplot(236)
plt.imshow(perspective_transform())

plt.show()
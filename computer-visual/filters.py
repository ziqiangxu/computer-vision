import cv2
import numpy as np


def stroke_edges(src, blur_k_size=7, edge_k_size=5):
    # 模糊函数，去噪声
    if blur_k_size >= 3:
        blurred_src = cv2.medianBlur(src, blur_k_size)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 边缘检测, 将结果直接写入gray_src
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_k_size)

    cv2.imshow("gray_src", gray_src)

    normalized_inverse_alpha = (1/255) * (255 - gray_src)

    cv2.imshow("nia", normalized_inverse_alpha)

    print(normalized_inverse_alpha)
    channels = cv2.split(src)
    # 对每个颜色通道都进行乘法运算
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    return cv2.merge(channels)


class VConvolutionFilter(object):
    """
    一般过滤器
    """
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel

    def apply(self, src: np.ndarray, dst: np.ndarray):
        cv2.filter2D(src, -1, self.kernel, dst)  # 第二个参数为负表示dst和src同样的位深度


class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        # kernel = np.full([3, 3], -1)
        # kernel[1, 1] = 9  锐化核
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.full([3, 3], -1)
        kernel[1, 1] = 8  # 边缘检测核
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.full([5, 5], 0.04)  # 半径为2的模糊核
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """
    会产生浮雕(emboss)/脊状(ridge)效果
    """
    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [0,   1, 2]])  # 不对称的特殊核
        VConvolutionFilter.__init__(self, kernel)


class Viewer(object):
    win_num = 0

    def __init__(self):
        pass

    def show_img(self, img: np.ndarray):
        name = "pic" + str(self.win_num)
        cv2.imshow(name, img)
        self.win_num += 1


def get_contours(img: np.ndarray):
    # img = np.zeros((200, 200), dtype=np.uint8)
    # img[50:150, 50:150] = 255
    if len(img.shape) != 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)  # 对图像像素值调整至127-255
    print(ret)
    viewer = Viewer()
    viewer.show_img(thresh)
    image, contours, hierarchy = cv2.findContours(thresh,
                                                  cv2.RETR_TREE,  # 层次类型
                                                  cv2.CHAIN_APPROX_SIMPLE)  # 轮廓逼近方法
    print("image:", image)
    print("contours:", contours)
    print("hierarchy:", hierarchy)

    color_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    viewer.show_img(color_img)
    img_gray = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    viewer.show_img(img_gray)


if __name__ == '__main__':
    file_path = "/home/xu/Pictures/cumt.jpg"
    file_path = "/home/xu/Pictures/timg.jpeg"
    file_path = "/home/xu/Pictures/cup.png"
    file_path = "/home/xu/ipynb/opencv/pic/dog.jpeg"
    img = cv2.imread(file_path)
    viewer = Viewer()

    # print(img.shape)
    # res = stroke_edges(img, 7, 5)
    # cv2.imshow('src', img)
    # cv2.imshow('dst', res)

    """filters test"""
    # src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = img
    dst = np.full(src.shape, 0, dtype=src.dtype)
    viewer.show_img(src)

    SharpenFilter().apply(src, dst)
    viewer.show_img(dst)

    FindEdgesFilter().apply(src, dst)
    viewer.show_img(dst)

    BlurFilter().apply(src, dst)
    viewer.show_img(dst)

    EmbossFilter().apply(src, dst)
    viewer.show_img(dst)

    """边缘检测"""
    canny = cv2.Canny(src, 100, 300)
    viewer.show_img(canny)

    """轮廓检测"""
    get_contours(src)

    cv2.waitKey(-1)  # 按任意键关闭所有窗口
    cv2.destroyAllWindows()


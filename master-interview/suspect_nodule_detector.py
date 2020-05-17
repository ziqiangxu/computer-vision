import numpy as np
import matplotlib.pyplot as plt
import tools
import cv2
import skimage.measure as measure
import scipy
import math
from const import *


def show_gray_img(image, tile=None):
    # todo save as log
    pass


def show_img(image, tile=None):
    # todo save as log
    pass


def get_suspected_nodule_mask_2d(slice_: np.ndarray, axis_spacing: float) -> np.ndarray:
    """
    window width 1000, window level -650, get this from a paper
    :param slice_:
    :param axis_spacing spacing of the (normal direction)
    :return:
    """
    window_start = -1150
    window_end = -150  # intensity of background almost all are 0 around
    slice_u8 = tools.get_window_img(slice_, (-1150, -150))
    show_gray_img(slice_u8, 'original')

    canny_th1 = 50
    canny_th2 = 200
    # canny_th1 = 200
    # canny_th2 = 500

    # gauss blur to down the noise
    slice_u8_blur = cv2.GaussianBlur(slice_u8, (3, 3), 0)
    show_gray_img(slice_u8_blur, 'blur_img')
    edges_blur = cv2.Canny(slice_u8_blur, canny_th1, canny_th2)
    show_gray_img(edges_blur, 'blur_edges')
    edges_blur[edges_blur == 255] = 1

    # invert the edges, todo why??? forget that
    edges_blur[edges_blur == 0] = 2
    edges_blur[edges_blur == 1] = 0
    edges_blur[edges_blur == 2] = 1
    # seperate = edges_blur * slice_u8

    separate_connected = measure.label(edges_blur, connectivity=1)

    pixel_num_min = SPACING_Z * SPACING_Z / SPACING_X / SPACING_Y
    pixel_num_max = math.pi * 15 * 15 / (SPACING_X * SPACING_X)

    # reserve 0 to mark the background
    separate_connected[separate_connected == 0] = separate_connected.max() + 1
    large_area_count = 0
    small_area_count = 0
    for v in np.unique(separate_connected):
        pixel_num = np.sum(separate_connected == v)
        if pixel_num < pixel_num_min:
            separate_connected[separate_connected == v] = 0
            small_area_count += 1
        elif pixel_num > pixel_num_max:
            separate_connected[separate_connected == v] = 0
            large_area_count += 1
    print(f'some area ignored, small area: {small_area_count}, large area: {large_area_count} ')
    show_img(separate_connected)
    show_gray_img(separate_connected)

    # morphology operate to erase small areas
    kernel = np.ones((3, 3), np.uint8)
    seperate_mask = separate_connected.astype(np.uint8)
    seperate_mask[seperate_mask != 0] = 1
    seperate_mask = cv2.morphologyEx(seperate_mask, cv2.MORPH_OPEN, kernel)
    show_gray_img(np.bitwise_and(separate_connected, seperate_mask))
    show_gray_img(separate_connected)
    show_gray_img(seperate_mask)

    # remove region diameter longer than 30mm
    # nodules and other tissues are mixture
    # https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
    # detected object less than before, don't matter, here we wanna drop big objects
    contours, hierarcy = cv2.findContours(seperate_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"number of contours: {len(contours)}")
    # print(len(contours), contours, hierarcy)

    img = cv2.cvtColor(slice_u8, cv2.COLOR_GRAY2BGRA)
    my_contours = list()
    # Get box for every contours
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        if max(rect[1]) > MAX_NODULE_SIZE / axis_spacing:
            o_x, o_y = rect[0]
            # remove this object by index, firstly, get the region id
            roi_id = separate_connected[int(o_x), int(o_y)]
            print(f'A large object detected: {max(rect[1])} pixels, at {o_x}, {o_y}, region_id: {roi_id}')
            separate_connected[separate_connected == roi_id] = 0
            break

        rect = cv2.boxPoints(rect)
        my_contours.append(cont)
        my_contours.append(rect.astype(np.int))
    #     my_contours = [rect.astype(np.int)]
    img = cv2.drawContours(img, my_contours, -1, (0, 255, 0), 2)
    show_img(img)
    return separate_connected


def filter_in_xy(slice_: np.ndarray):
    """
    Resize to 512, 512 then OPEN
    :param slice_:
    :return:
    """
    init_shape = slice_.shape[1], slice_.shape[0]
    slice_ = cv2.resize(slice_, (512, 512))
    mask = slice_.copy()
    mask[mask != 0] = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    slice_ = cv2.resize(slice_, init_shape)
    mask = cv2.resize(mask, init_shape)
    return slice_ * mask


def get_suspected_nodule_atlas(images: np.ndarray) -> np.ndarray:
    """
    Get nodule atlas
    :param images: lung images, and have no other organ, Z axis as the first chanel
    :return:
    """
    # Along the Z axis
    i = 0
    atlas = np.zeros(images.shape)
    for slice_ in images:
        slice_atlas = get_suspected_nodule_mask_2d(slice_, SPACING_Z)
        atlas[i] = slice_atlas
        i += 1

    # X and Y, morphology operate: OPEN to erase discontinuous pixel
    for i in range(images.shape[1]):
        slice_ = images[:, i, :]
        images[:, i, :] = filter_in_xy(slice_)

    for i in range(images.shape[2]):
        slice_ = images[:, :, i]
        images[:, :, i] = filter_in_xy(slice_)
    return atlas


def debug_get_suspected_nodule_atlas():
    volume = tools.get_arr_from_nii('data/output/lung-roi.nii')
    res = get_suspected_nodule_atlas(volume)
    tools.npy2nii(res, 'data/output/suspect-nodules.nii')


if __name__ == '__main__':
    debug_get_suspected_nodule_atlas()
    pass

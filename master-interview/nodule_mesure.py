import numpy as np
import matplotlib.pyplot as plt
import tools
import cv2
import skimage.measure as measure
from const import *


def get_neighbor(current_slice: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
    """
    :param current_slice:
    :param neighbor:
    :return:
    """
    label = measure.label(neighbor, connectivity=1)
    unused_type = label.max() + 2
    for p in np.argwhere(current_slice == 1):
        label_copy = label.copy()
        if neighbor[p[0], p[1]] >= 1:
            current_type = label_copy[p[0], p[1]]
            label_copy[label_copy != current_type] = unused_type
            label_copy[label_copy == current_type] = 1
            label_copy[label_copy == unused_type] = 0
            # add to neighbor
            neighbor += label_copy.astype(np.uint8)
            # break
            continue
    neighbor[neighbor != 0] = 1
    return neighbor


def recognize_nodule(nodule):
    """
    todo judge whether a really nodule got
    :param nodule:
    :return:
    """
    threshold = 0.7
    nodule[nodule > threshold] = 1
    nodule[nodule <= threshold] = 0

    nodule = nodule.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    # x
    for x in range(nodule.shape[0]):
        nodule[x] = cv2.morphologyEx(nodule[x], cv2.MORPH_OPEN, kernel)

    for y in range(nodule.shape[1]):
        nodule[:, x, :] = cv2.morphologyEx(nodule[:, x, :], cv2.MORPH_OPEN, kernel)

    for z in range(nodule.shape[2]):
        nodule[:, :, z] = cv2.morphologyEx(nodule[:, :, z], cv2.MORPH_OPEN, kernel)

    for x in range(nodule.shape[0]):
        nodule[x] = cv2.morphologyEx(nodule[x], cv2.MORPH_CLOSE, kernel)

    for y in range(nodule.shape[1]):
        nodule[:, x, :] = cv2.morphologyEx(nodule[:, x, :], cv2.MORPH_CLOSE, kernel)

    for z in range(nodule.shape[2]):
        nodule[:, :, z] = cv2.morphologyEx(nodule[:, :, z], cv2.MORPH_CLOSE, kernel)

    nodule_atlas = measure.label(nodule, neighbors=4)
    tools.npy2nii(nodule_atlas.astype(np.uint8), 'data/log/nod-atlas1.nii')

    # Mark the nodule pixels by 1, the rest fills by 0

    tools.npy2nii(nodule_atlas.astype(np.uint8), 'data/log/nod-atlas2.nii')

    center = np.array(nodule.shape) / 2
    center = center.astype(np.int8)
    center_x, center_y, center_z = tuple(center)
    # presume that the center pixel belongs to nodule
    center_type = nodule_atlas[center_x, center_y, center_z]
    unused_value = nodule_atlas.max() + 2
    nodule_atlas[nodule_atlas == center_type] = unused_value
    nodule_atlas[nodule_atlas != unused_value] = 0
    nodule_atlas[nodule_atlas == unused_value] = 1
    return nodule_atlas.astype(np.uint8)

    # A better api found
    # connect all pixels belong to nodule (3D)
    # slice_index = center_x
    # label = measure.label(nodule[slice_index], connectivity=1)
    # init_type = label[center_y, center_z]
    # unused_type = label.max() + 2
    # label[label == init_type] = unused_type
    # label[label != init_type] = 0
    # label[label == unused_type] = 1
    # current_slice = label
    #
    # where = 0
    # while slice_index > 0:
    #     slice_index -= 1
    #     where += 1
    #     nodule[slice_index] = get_neighbor(current_slice, nodule[slice_index])
    #     plt.imshow(nodule[slice_index])
    #     plt.waitforbuttonpress()
    #     current_slice = nodule[slice_index]
    # while slice_index < nodule.shape[0] - 1:
    #     slice_index += 1
    #     nodule[slice_index] = get_neighbor(current_slice, nodule[slice_index])
    #     current_slice = nodule[slice_index]
    # return nodule


def get_none_zero_pixel_num(map_: np.ndarray):
    """
    Get none zero pixel num
    :param map_:
    :return:
    """
    map_[map_ != 0] = 1
    return map_.sum()


def get_long_short_axis(map_: np.ndarray, axis=''):
    """
    Firstly, make the three resolution identity
    :param map_: 
    :param axis: x, y, z
    :return: 
    """
    map_[map_ != 0] = 1
    map_ = map_.astype(np.uint8)
    size_raw = map_.shape
    if axis == 'x':
        size_new = (size_raw[0], int(size_raw[1] * (SPACING_Z / SPACING_Y)))
        img = cv2.resize(map_, size_new)
    elif axis == 'y':
        size_new = (size_raw[0], int(size_raw[1] * (SPACING_Z / SPACING_X)))
        img = cv2.resize(map_, size_new)
    else:
        # Along the Z axis, no need to resize
        img = map_
    pixel_length = SPACING_X
    img = img.astype(np.uint8)
    contours, hir = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dst_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst_img = cv2.drawContours(dst_img, contours, 0, (0, 255, 0), 2)
    rect = cv2.minAreaRect(contours[0])
    # plt.imshow(dst_img)
    # plt.waitforbuttonpress(-1)
    return rect[1]


def measure_nod(nod_atlas: np.ndarray):
    """
    Measure the four factor
    :param nod_atlas: 
    :return: 
    """
    volume = nod_atlas.sum() * SPACING_Z * SPACING_Y * SPACING_X
    map_x = nod_atlas.sum(0)
    map_y = nod_atlas.sum(1)
    map_z = nod_atlas.sum(2)
    area_x = SPACING_Z * SPACING_Y * get_none_zero_pixel_num(map_x)
    area_y = SPACING_X * SPACING_Z * get_none_zero_pixel_num(map_y)
    area_z = SPACING_X * SPACING_Y * get_none_zero_pixel_num(map_z)
    surface_area = 2 * (area_x + area_y + area_z)
    specific_surface_area = surface_area / volume

    # plt.imshow(map_x)
    # plt.imshow(map_y)
    # plt.imshow(map_z)
    # plt.waitforbuttonpress(-1)

    axis_x = get_long_short_axis(map_x, 'x')
    axis_y = get_long_short_axis(map_y, 'y')
    axis_z = get_long_short_axis(map_z, 'z')
    axis = []
    axis.extend(axis_x), axis.extend(axis_y), axis.extend(axis_z)
    long_axis = max(axis)
    short_axis = min(axis)

    return {
        "volume": volume,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "specific_surface_area": specific_surface_area
    }


def my_measure_nod():
    nod = tools.get_arr_from_nii("data/nod.nii")
    nod_atlas = recognize_nodule(nod)
    tools.npy2nii(nod_atlas, 'data/output/nod-atlas.nii')

    print(measure_nod(nod_atlas))


if __name__ == '__main__':
    my_measure_nod()

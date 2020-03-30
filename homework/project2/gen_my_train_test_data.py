import cv2 as cv
import numpy as np
import os
from typing import List
import logging


def gen_keypoints(kp_raw: np.float32)-> List[cv.KeyPoint]:
    """
    根据数组生成关键点列表
    """
    kp_arr = kp_raw.reshape(-1, 2)
    kps = []
    for p in kp_arr:
        kps.append(cv.KeyPoint(p[0], p[1], 0))
    return kps

def zoom_rect(
    point_left_top: np.ndarray, 
    point_right_bottom: np.ndarray, 
    size: tuple, zoom_factor=1.25
    ):
    """
    缩放矩形框
    """
    assert zoom_factor > 0
    zoom_factor -= 1
    rect_size = point_right_bottom - point_left_top
    
    plt = point_left_top - rect_size * zoom_factor / 2 
    plt[plt < 0] = 0
    
    prb = point_right_bottom + rect_size * zoom_factor / 2
    if prb[0] > size[0]:
        prb[0] = size[0]
    if prb[1] > size[1]:
        prb[1] = size[1]
    
    return (
        tuple(plt.astype(np.int32).tolist()),
        tuple(prb.astype(np.int32).tolist())
    )

def relative_keypoints(
    kp_raw: np.ndarray, 
    point_left_top: np.ndarray) -> np.ndarray:
    """
    return numpy array: (-1, 2)
    """
    kp_raw.reshape((-1, 2))
    return kp_raw - point_left_top

def gen_test_data(filepath, data, target_dir, mode="draw"):
    """
    @param mode: draw or crop
    """
    img = cv.imread(filepath)
    
    # 框出面部
    rec_point1 = (int(data[0]), int(data[1]))
    rec_point2 = (int(data[2]), int(data[3]))
    # cv图片的shape -->（高， 宽， 通道数）
    point_left_top, point_right_bottom = zoom_rect(np.int32(rec_point1),
                                                   np.int32(rec_point2), 
                                                   (img.shape[1], img.shape[0]),
                                                   1.25)
    
    print(f'file: {filepath}, p1: {rec_point1}, p2: {rec_point2}')
    

    face_region = img[point_left_top[1]:point_right_bottom[1],
                      point_left_top[0]:point_right_bottom[0]]
    
    # keypoints
    kp_raw = np.float32(data[4:]).reshape(-1, 2)
    new_kp_raw = kp_raw - np.int32(point_left_top)
    
    if mode == 'draw':
        kps = gen_keypoints(new_kp_raw)
#         print(kps)
        cv.drawKeypoints(face_region, kps, face_region)
        cv.imwrite(os.path.basename(filepath), face_region)
        
    # save face crop and keypoint from dataset
    n = 0
    target_name = f'{n}-{os.path.basename(filepath)}'
    target_path = os.path.join(target_dir, target_name)
    while os.path.exists(target_path):
        n += 1
        target_name = f'{n}-{os.path.basename(filepath)}'
        target_path = os.path.join(target_dir, target_name)
    cv.imwrite(target_path, face_region)

# 数据准备，截取面部区域
# with open('data/I/label.txt', 'r') as f:
#     # todo clean the directory where store the train data
#     for i in range(10):
#         line = f.readline()
# #         print(line)
#         label = line[:-1].split(' ') # remove the line break
#         data = []
#         for j in label[1:]:
#             data.append(float(j))
#         filepath = os.path.join('data/I', label[0])
#         gen_test_data(filepath, data, "mydata/train/")

def gen_data_label(data_raw: list, file_path: str, draw_face=False) -> str:
    data = []
    for i in data_raw:
        data.append(float(i))
    img = cv.imread(file_path)
    # 扩展面部区域
    point_left_top, point_right_bottom = zoom_rect(
        np.int32(data[0:2]),
        np.int32(data[2:4]),
        (img.shape[1], img.shape[0])
    )
    # 面部关键点相对坐标
    kps = np.float32(data[4:]).reshape(-1, 2)
    relative_kps = relative_keypoints(kps, np.float32(point_left_top))
    relative_kps = relative_kps.flatten()
    
    # 生成字符串
    res = f'{file_path} {point_left_top[0]} {point_left_top[1]} {point_right_bottom[0]} {point_right_bottom[1]} '
    for i in relative_kps:
        res += f'{i} '
    if draw_face:
        face_region = img[point_left_top[1]:point_right_bottom[1],
                        point_left_top[0]:point_right_bottom[0]]
        kps_cv = gen_keypoints(relative_kps)
        cv.drawKeypoints(face_region, kps_cv, face_region)
        filename = os.path.basename(file_path)
        cv.imwrite(filename, face_region)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # 总共2000张图片
    test_size = 400
    train_size = 1600
    valid_face_num = 0

    train_file = open('train.txt', 'w')

    with open('data/I/label.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
        # for i in range(10):
            data_raw = lines[i].split(' ')
            file_path = f'data/I/{data_raw[0]}'
            if os.path.exists(file_path):
                valid_face_num += 1
                res = gen_data_label(data_raw[1:], file_path)
                train_file.write(res + '\n')

    with open('data/II/label.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
        # for i in range(10):
            data_raw = lines[i].split(' ')
            file_path = f'data/II/{data_raw[0]}'
            if os.path.exists(file_path):
                valid_face_num += 1
                res = gen_data_label(data_raw[1:], file_path)
                train_file.write(res + '\n')

    train_file.close()
    logging.info(f'total valid faces: {valid_face_num}')
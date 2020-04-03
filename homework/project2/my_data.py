from typing import List

import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms

TRAIN_BOARDER = 112


def get_data(normalize=True) -> tuple:
    with open('train.txt', 'r') as f:
        lines = f.readlines()
    faces_num = len(lines)
    # 2 valid 8 train
    valid_num = faces_num // 5

    if normalize:
        tsfm = transforms.Compose([
            Normalize(), ToTensor()
            ])
    else:
        tsfm = transforms.Compose([ToTensor()])

    return (FaceLandmarksDataset(lines[valid_num:], tsfm),
            FaceLandmarksDataset(lines[:valid_num], tsfm))


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    # 矩形框需要是整数，先将字符串转为浮点数，再转为整数
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


def channel_norm(img: np.ndarray) -> np.ndarray:
    """
    对图片进行normalize
    :param img:
    :return:
    """
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # 具体的重采样方法呢？
        # image = Image.fromarray(image)
        # image_resize = np.asarray(
        #                     image.resize((TRAIN_BOARDER, TRAIN_BOARDER), Image.BILINEAR),
        #                     dtype=np.float32)       # Image.ANTIALIAS)
        image_resize = np.asarray(
            cv.resize(image, (TRAIN_BOARDER, TRAIN_BOARDER)),
            dtype=np.float32
            )
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }


def gen_keypoints(kp_raw: np.float32)-> List[cv.KeyPoint]:
    """
    Get key points from arr: [x1, y1, x2, y2, ...]
    :param kp_raw:
    :return:
    """
    kp_arr = kp_raw.reshape(-1, 2)
    kps = []
    for p in kp_arr:
        kps.append(cv.KeyPoint(p[0], p[1], 0))
    return kps


def input_from_image(filename: str) -> torch.Tensor:
    """
    Get input tensor from file
    :param filename:
    :return:
    """
    img = cv.imread(filename, 0)
    normalizer = Normalize()
    data = {
        'image': img,
        'landmarks': np.array([])
    }
    data = normalizer.__call__(data)
    # to tensor
    to_tensor = ToTensor()
    input_data = to_tensor.__call__(data)
    return input_data


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines: List[str],
                 transform: transforms.Compose = None):
        """
        :param src_lines: src_lines
        :param transform: data transform
        """
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        # img = Image.open(img_name).convert('L')
        img = cv.imread(img_name, 0)
        # img = Image.open(img_name)

        # img_crop = img.crop(tuple(rect))
        img_crop = img[rect[1]:rect[3], rect[0]:rect[2]]
        # cv.imwrite('crop.png', img_crop)
        landmarks = np.array(landmarks).astype(np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:

        # img_crop = np.asarray(
        #                     img_crop.resize((TRAIN_BOARDER, TRAIN_BOARDER), Image.BILINEAR),
        #                     dtype=np.float32)       # Image.ANTIALIAS)
        img_crop = np.asarray(
            cv.resize(img_crop, (TRAIN_BOARDER, TRAIN_BOARDER)),
            dtype=np.float32
            )
        # 获取缩放的系数，对关键点坐标进行变换
        zoom_x = TRAIN_BOARDER / (rect[2] - rect[0])
        zoom_y = TRAIN_BOARDER / (rect[3] - rect[1])
        assert zoom_x > 0 and zoom_y > 0
        landmarks: np.ndarray = landmarks.reshape(-1, 2)

        # x坐标变换
        landmarks[:, 0] = landmarks[:, 0] * zoom_x
        # y坐标变换
        landmarks[:, 1] = landmarks[:, 1] * zoom_y
        landmarks = landmarks.reshape(-1)
        # end my code

        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    train_set, _ = get_data(False)

    sample = train_set[0]
    img = sample['image']
    landmarks = sample['landmarks']

    # img = np.random.random((100, 100))
    crop_img = np.copy(img[0])
    crop_img = np.transpose(crop_img, (1, 2, 0))
    crop_img = np.copy(crop_img.astype(np.uint8))
    # cv.imwrite('log.png', gbr_img)

    # keypoints
    kp_raw = landmarks.reshape(-1, 2)
    kps = []
    for p in kp_raw:
        cv.KeyPoint()
        kps.append(cv.KeyPoint(p[0], p[1], 0))

    cv.drawKeypoints(crop_img, kps, crop_img)
    cv.imwrite('log.png', crop_img)

    # plt.waitforbuttonpress()

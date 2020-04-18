from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

from const import *
import cv2 as cv

TEST_LABEL = 'traffic-sign/test_label.csv'
TRAIN_LABEL = 'traffic-sign/train_label.csv'


def parse_line(line: str):
    """
    Format of the line
    Sequence number, file path, category
    :param line:
    :return:
    """
    no, file_path, type_idx = line.split(',')
    return file_path, int(type_idx)


class Normalize(object):
    """
        Resize to train_boarder x train_boarder.
        Then do channel normalization: (image - mean) / std_variation
    """
    def channel_norm(self, img: np.ndarray) -> np.ndarray:
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

    def __call__(self, sample):
        """
        :param sample:
        :return:
        """
        image, category = sample['image'], sample['category']
        image_resize = np.asarray(
            cv.resize(image, (TRAIN_BOARDER, TRAIN_BOARDER)),
            dtype=np.float32
            )
        image = self.channel_norm(image_resize)
        return {'image': image,
                'category': category}


class ToTensor(object):
    def __call__(self, sample):
        image, category = sample['image'], sample['category']
        image = np.expand_dims(image, axis=0)
        return {
            'image': torch.from_numpy(image),
            'category': torch.from_numpy(category)
        }


class DataElement:
    """
    todo replace dict{'image': image, 'category': category} with this class
    """
    def __init__(self, image: np.ndarray, category: int):
        self.image = image
        self.category = category


class TrafficSignDataset(Dataset):
    def __init__(self, src_lines: List[str],
                 transform: transforms.Compose = None):
        """
        :param src_lines: src_lines
        :param transform: data transform, like ToTensor or Normalize
            (which inherit from transforms.Compose???)
        """
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx: int):
        img_name, type_idx = parse_line(self.lines[idx])
        # image
        # img = Image.open(img_name).convert('L')
        image = cv.imread(img_name, 0)
        category = np.zeros(CATEGORY_NUM)
        category[type_idx] = 1.0

        sample = {'image': image, 'category': category}
        sample = self.transform(sample)
        return sample


def get_data():
    trans = transforms.Compose([Normalize(), ToTensor()])
    with open(TRAIN_LABEL, 'r') as f:
        f.readline()
        train_dataset = TrafficSignDataset(f.readlines(), trans)
    with open(TEST_LABEL, 'r') as f:
        f.readline()
        test_dataset = TrafficSignDataset(f.readlines(), trans)
    return train_dataset, test_dataset


if __name__ == '__main__':
    with open(TEST_LABEL, 'r') as f:
        # the first line is header of the table
        print('header of the table:', f.readline())
        for i in range(10):
            print(parse_line(f.readline()))
    train_dataset, test_dataset = get_data()
    b = train_dataset[0]
    print(b)

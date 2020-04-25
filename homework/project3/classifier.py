import argparse
from typing import Any

import torch
import torchvision
import torch.optim as optim
from torch.nn.modules.module import T_co
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from data import get_data
from const import *

torch.set_default_tensor_type(torch.FloatTensor)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # backbone nx3x112x112
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_features = vgg16.features

        # conv1_1 block: nx512x3x3 -> nx128x3x3
        self.conv1_1 = nn.Conv2d(512, 128, 3, 1, padding=1)
        # active
        self.relu1 = nn.PReLU()

        # conv1_2 block: nx128x3x3 -> nx62x1x1
        self.conv1_2 = nn.Conv2d(128, CATEGORY_NUM, 3, 1, padding=0)
        # softmax
        self.soft_max = nn.Softmax(1)

    def forward(self, x) -> T_co:
        x = self.vgg16_features(x)

        x = self.conv1_1(x)
        x = self.relu1(x)

        x = self.conv1_2(x)
        x = self.soft_max(x)
        return x.view(-1, CATEGORY_NUM)


def train():
    pass


class Argument:
    def __init__(self, arguments):
        self.batch_size = arguments.batch_size
        self.test_batch_size = arguments.test_batch_size
        self.epoch = arguments.epoch
        self.lr = arguments.lr
        self.momentum = arguments.momentum
        self.seed = arguments.seed
        self.log_interval = arguments.log_interval
        self.save_model = arguments.save_model
        self.save_directory = arguments.save_directory
        self.phase = arguments.phase
        self.model = arguments.model
        self.input = arguments.input


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classifier')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    """
    store_true' and 'store_false' - These are special cases of 'store_const' using for storing the values True and False 
    respectively. In addition, they create default values of False and True respectively. 
    """
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_false') # ??true
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')

    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--model', type=str, default='trained_models/newest.pt',
                        help='the model to load')
    parser.add_argument('--input', type=str, help='path of input image')
    return Argument(parser.parse_args())


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(args.seed)

    device = torch.device('cuda')
    model = Net().to(device)
    criterion_pts = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), )

    train_data, test_data = get_data()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import argparse
import my_data as data
import os
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.FloatTensor)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)

        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.ave_pool(self.prelu1_1(self.conv1_1(x)))
        # print('x after block1 and pool shape should be 32x8x27x27: ', x.shape)     # good
        # block 2
        x = self.prelu2_1(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be 32x16x25x25: ', x.shape) # good
        x = self.prelu2_2(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu shape should be 32x16x23x23: ', x.shape) # good
        x = self.ave_pool(x)
        # print('x after block2 and pool shape should be 32x16x12x12: ', x.shape)
        # block 3
        x = self.prelu3_1(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be 32x24x10x10: ', x.shape)
        x = self.prelu3_2(self.conv3_2(x))
        # print('b3: after conv3_2 and pool shape should be 32x24x8x8: ', x.shape)
        x = self.ave_pool(x)
        # print('x after block3 and pool shape should be 32x24x4x4: ', x.shape)
        # block 4
        x = self.prelu4_1(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be 32x40x4x4: ', x.shape)

        # points branch
        ip3 = self.prelu4_2(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be 32x80x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # print('ip3 flatten shape should be 32x1280: ', ip3.shape)
        ip3 = self.preluip1(self.ip1(ip3))
        # print('ip3 after ip1 shape should be 32x128: ', ip3.shape)
        ip3 = self.preluip2(self.ip2(ip3))
        # print('ip3 after ip2 shape should be 32x128: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be 32x42: ', ip3.shape)
        
        return ip3


def train(
    args, train_data_loader: DataLoader, valid_data_loader: DataLoader, model: Net,
    criterion, optimizer: optim.Optimizer, device
):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    epochs = args.epochs
    
    train_losses = []
    valid_losses = []
    for epoch_id in range(epochs):
        # train_loss = 0.0
        # valid_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        train_batch_cnt = 0
        train_mean_pts_loss = 0.0
        for batch_idx, batch in enumerate(train_data_loader):
            train_batch_cnt += 1
            img = batch['image']
            landmark = batch['landmarks']

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables(torch.Tensor)
            optimizer.zero_grad()
            output_pts = model(input_img)
            loss = criterion(output_pts, target_pts)
            
            # do BP automatically
            loss.backward()
            optimizer.step()  # 更新优化器中的参数
            train_mean_pts_loss += loss.item()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
        train_mean_pts_loss /= train_batch_cnt
        train_losses.append(train_mean_pts_loss)

        #######################
        # validate the model #
        #######################
        valid_mean_pts_loss = 0.0
        model.eval() #prepare model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for valid_batch_idx, batch in enumerate(valid_data_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)
                valid_loss = criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            valid_losses.append(valid_mean_pts_loss)
    
            print('Valid: pts_loss: {:.6f}'.format(
                    valid_mean_pts_loss
                )
            )
        print('====================================================')
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 
            # f'detector_epoch_{epoch_id}_{train_mean_pts_loss}_{valid_mean_pts_loss}.pt')
            f'detector_epoch_{epoch_id}.pt')
            torch.save(model.state_dict(), saved_model_name)
        draw_loss(train_losses, valid_losses, args.phase)

    return train_losses, valid_losses


def model_test(model: Net, test_dataset, device, criterion):
    """

    :param model:
    :param test_dataset:
    :param device:
    :param criterion:
    :return:
    """
    mean_loss = 0
    model.eval()
    batch_cnt = 0
    for batch_idx, batch in enumerate(test_dataset):
        img: torch.Tensor = batch['image']
        landmarks: torch.Tensor = batch['landmarks']
        img = img.to(device)
        landmarks = landmarks.to(device)

        output = model.forward(img)
        loss = criterion(output, landmarks)
        mean_loss += loss.item()
        batch_cnt += 1
        print(f'loss in batch {batch_idx}: {loss.item()}')
    mean_loss = mean_loss / batch_cnt
    return mean_loss


def draw_loss(train_losses, valid_losses, prefix):
    plt.cla()
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train_losses')
    ax.plot(valid_losses, label='valid_losses')
    ax.legend()
    fig.savefig(f'log/{prefix}_losses.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    """
    store_true' and 'store_false' - These are special cases of 'store_const' using for storing the values True and False 
    respectively. In addition, they create default values of False and True respectively. 
    """
    parser.add_argument('--no-cuda', action='store_true') # 默认false
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_false') # 默认true
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')

    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--load-model', type=str, default='trained_models/newest.pt',
                        help='the model to load')
    parser.add_argument('--input', type=str, help='path of input image')
    parser.add_argument('--output', type=str, help='path of out image')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    #是否使用GPU
    # For single GPU
    use_cude = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cude else 'cpu')
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cude else {}

    # get train data 
    train_set, test_set = data.get_data()

    print("===> loading datasets")
    batch_size = args.batch_size
    if args.phase != 'Predict' and args.phase != 'predict':
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size
        )

    print("===> building model")
    model: Net = Net().to(device)
    criterion_pts = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)

    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        # how to do test?
        test_loader = valid_loader
        saved_state = torch.load(args.load_model)
        model.load_state_dict(saved_state)
        test_loss = model_test(model, test_loader, device, criterion_pts)
        print(f'test loss: {test_loss}')
        print('====================================================')
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        # how to do finetune?
        # load the saved model, and adjust the parameters
        saved_state = torch.load('trained_models/detector_epoch_499.pt')
        model.load_state_dict(saved_state)

        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('====================================================')
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        import cv2 as cv
        import numpy as np
        # how to do predict?
        # 1. load the model
        # 2. call forward
        saved_state = torch.load(args.load_model)
        model.load_state_dict(saved_state)
        # predict data from validation dataset
        model.eval()

        # for i in range(10):
        # filename = f'mydata/face-new{i}.png'
        filename = args.input
        out_img = args.output
        if not os.path.exists(filename):
            raise Exception(f'{filename} not found')
        input_data = data.input_from_image(filename)

        img: torch.Tensor = input_data['image']
        img = img.unsqueeze(0)

        output: torch.Tensor = model.forward(img.to(device))
        print(output)

        output_cpu = output.cpu()
        kps_raw = output_cpu[0].detach().numpy()
        kps_raw = kps_raw.reshape(-1, 2)

        # Scale the key points
        img_color = cv.imread(filename)
        h_factor = img_color.shape[0] / data.TRAIN_BOARDER
        w_factor = img_color.shape[1] / data.TRAIN_BOARDER
        # print(img_color.shape, w_factor, h_factor)
        # print('previous: ', kps_raw.reshape(-1,21))
        kps_raw[:, 0] = kps_raw[:, 0] * w_factor
        kps_raw[:, 1] = kps_raw[:, 1] * h_factor
        kps = data.gen_keypoints(kps_raw)
        # print('zoomed:', kps_raw.reshape(-1,21))

        cv.drawKeypoints(img_color, kps, img_color, color=(0, 0, 255))
        # cv.imwrite(f'log/{os.path.basename(filename)}', img)
        cv.imwrite(out_img, img_color)

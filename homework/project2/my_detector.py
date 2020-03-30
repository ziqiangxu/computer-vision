import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.Dataloader as DataLoader
import argparse
import homework.project2.my_data as data
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


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
            landmark = batch['landmark']

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
            f'detector_epoch_{epoch_id}.pt')
            torch.save(model.state_dict, saved_model_name)

    return train_losses, valid_losses

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
    parser.add_argument('--no-cuda', action='store_false')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true')

    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    #是否使用GPU
    # For single GPU
    use_cude = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cude else 'cpu')
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cude else {}

    # get train data 
    train_set, test_set = data.get_train_test_set()

    print("loading datasets")
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size
    )

    print("building model")
    model = Net().to(device)
    criterion_pts = nn.MSELoss()
    optimizer = optim.SGD(model.parameters, args.lr, args.momentum)

    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        # how to do test?
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        # how to do finetune?
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?

    

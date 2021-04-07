import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import glob
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_starts", default=10)
    parser.add_argument("--lr", type=float) #specify learning rate
    parser.add_argument("--rw", type=float) #specify regularization weight
    parser.add_argument("--n_iter", type=int) #specify number of iterations
    parser.add_argument("--do", type=float) #specify drop-out
    args = parser.parse_args()

    lr = args.lr
    rw = args.rw
    drop_rate = args.do
    n_starts = args.n_starts
    n_iter = args.n_iter

    print(n_starts)

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(8464, 120)  # 6*6 from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x, p=0):
            drop2d = nn.Dropout2d(p)
            drop1d = nn.Dropout(p)

            # Max pooling over a (2, 2) window
            x = self.pool(drop2d(F.relu(self.conv1(x))))
            # If the size is a square you can only specify a single number
            x = self.pool(drop2d(F.relu(self.conv2(x))))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = drop1d(x)
            x = F.relu(self.fc2(x))
            x = drop1d(x)
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features


    def calc_loss(pred, obs):
        loss = nn.MSELoss()
        return loss(pred, obs)


    def sortKeyFunc(s):
        return int(os.path.basename(s)[4:-4])


    def load_dataset(folder):
        files = glob.glob('../data/' + folder + '/*.csv')
        files.sort(key=sortKeyFunc)  # glob returns list with arbitrary order

        l = len(files)
        dataset = np.zeros((l, 1000, 99))
        for k, file in enumerate(files):
            cell = np.genfromtxt(file, delimiter=',')
            dataset[k, :, :] = cell  # flip voltage dimension

        return dataset

    #Import data
    #Input data
    train_data = load_dataset('train')
    test1_data = load_dataset('test1')
    test2_data = load_dataset('test2')

    #Output data
    train_out = torch.from_numpy(np.log10(
        np.genfromtxt('../data/cycle_lives/train_cycle_lives.csv'))).float()
    test1_out = torch.from_numpy(np.log10(
        np.genfromtxt('../data/cycle_lives/test1_cycle_lives.csv'))).float()
    test2_out = torch.from_numpy(np.log10(
        np.genfromtxt('../data/cycle_lives/test2_cycle_lives.csv'))).float()

    N, V, C = train_data.shape  # number of batteries, #number of voltage points, number of cycles

    train_data = torch.from_numpy(train_data[:, ::10, :] - train_data[:, ::10, 8][:, :, np.newaxis]).float()
    test1_data = torch.from_numpy(test1_data[:, ::10, :] - test1_data[:, ::10, 8][:, :, np.newaxis]).float()
    test2_data = torch.from_numpy(test2_data[:, ::10, :] - test2_data[:, ::10, 8][:, :, np.newaxis]).float()

    # check for outliers and set values equal to zero
    train_data[np.where(train_data > 1)] = 0
    test1_data[np.where(test1_data > 1)] = 0
    test2_data[np.where(test2_data > 1)] = 0

    train_data[np.where(train_data < -0.2)] = 0
    test1_data[np.where(test1_data < -0.2)] = 0
    test2_data[np.where(test2_data < -0.2)] = 0

    #rescale input data
    stdevs = torch.std(train_data)
    train_data = (train_data) / stdevs
    test1_data = (test1_data) / stdevs
    test2_data = (test2_data) / stdevs

    #rescale output data
    out_means = torch.mean(train_out)
    train_out = train_out - out_means
    test1_out = test1_out - out_means
    test2_out = test2_out - out_means

    train_err_stor = []
    test1_err_stor = []
    test2_err_stor = []

    train_pred_stor = []
    test1_pred_stor = []
    test2_pred_stor = []

    loss_stor = []

    for _ in range(n_starts):
        net = Net()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        store_loss = []
        best_valid = 1e10

        #pretty_print('step', 'train MSE')
        for step in range(n_iter):

            preds = net(train_data[:, None, :, :], p=drop_rate)
            train_err = calc_loss(preds.squeeze(), train_out)

            weight_norm = torch.tensor(0.)
            for w in net.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_err.clone()
            loss += rw * weight_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            store_loss.append(loss.item())


        train_pred = 10 ** (net(train_data[:, None, :, :]) + out_means).detach().squeeze()
        train_pred_stor.append(train_pred)
        train_err_stor.append(torch.sqrt(
            calc_loss(train_pred,10 ** (train_out + out_means))))

        test1_data_no = torch.cat([test1_data[:21], test1_data[22:]])
        test1_out_no = torch.cat([test1_out[:21], test1_out[22:]])

        test1_pred = 10 ** (net(test1_data_no[:, None, :, :]) + out_means).detach().squeeze()
        test1_pred_stor.append(test1_pred)
        test1_err_stor.append(torch.sqrt(calc_loss(test1_pred,10 ** (test1_out_no + out_means))))

        test2_pred = 10 ** (net(test2_data[:, None, :, :]) + out_means).detach().squeeze()
        test2_pred_stor.append(test2_pred)
        test2_err_stor.append(torch.sqrt(calc_loss(test2_pred, 10 ** (test2_out + out_means))))

        loss_stor.append(store_loss)

    filename = 'CNN_n' + str(n_iter) + '_rw' + str(rw) + '_lr' + str(lr) + '_do' + str(drop_rate) + '.pt'

    torch.save({'train_pred': train_pred_stor, 'train_err': train_err_stor, 'test1_pred': test1_pred_stor,
                'test1_err': test1_err_stor, 'test2_pred': test2_pred_stor, 'test2_err': test2_err_stor,
                'loss': loss_stor}, filename)

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
    parser.add_argument("--hd", type=int) #specify the hidden dimension
    args = parser.parse_args()

    lr = args.lr
    rw = args.rw
    hd = args.hd
    n_starts = args.n_starts
    n_iter = args.n_iter


    class MLP(nn.Module):
        def __init__(self, hidden_dim=256):
            super(MLP, self).__init__()
            lin1 = nn.Linear(1000, hidden_dim)
            lin2 = nn.Linear(hidden_dim, hidden_dim)
            lin3 = nn.Linear(hidden_dim, 1)
            # add this for reproducibility
            # torch.manual_seed(0)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

        def forward(self, input):
            out = input.view(input.shape[0], V)
            out = self._main(out)
            return out


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
    
    train_data = torch.from_numpy(train_data[:, :, 98] - train_data[:, :, 8]).float()
    test1_data = torch.from_numpy(test1_data[:, :, 98] - test1_data[:, :, 8]).float()
    test2_data = torch.from_numpy(test2_data[:, :, 98] - test2_data[:, :, 8]).float()

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
        net = MLP(hidden_dim=hd)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        store_loss = []
        best_valid = 1e10

        #pretty_print('step', 'train MSE')
        for step in range(n_iter):

            preds = net(train_data)
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


        train_pred = 10 ** (net(train_data) + out_means).detach().squeeze()
        train_pred_stor.append(train_pred)
        train_err_stor.append(torch.sqrt(
            calc_loss(train_pred,10 ** (train_out + out_means))))

        test1_data_no = torch.cat([test1_data[:21], test1_data[22:]])
        test1_out_no = torch.cat([test1_out[:21], test1_out[22:]])

        test1_pred = 10 ** (net(test1_data_no) + out_means).detach().squeeze()
        test1_pred_stor.append(test1_pred)
        test1_err_stor.append(torch.sqrt(calc_loss(test1_pred,10 ** (test1_out_no + out_means))))

        test2_pred = 10 ** (net(test2_data) + out_means).detach().squeeze()
        test2_pred_stor.append(test2_pred)
        test2_err_stor.append(torch.sqrt(calc_loss(test2_pred, 10 ** (test2_out + out_means))))

        loss_stor.append(store_loss)

    filename = 'MLP_n' + str(n_iter) + '_rw' + str(rw) + '_lr' + str(lr) + '_hd' + str(hd) + '.pt'

    torch.save({'train_pred': train_pred_stor, 'train_err': train_err_stor, 'test1_pred': test1_pred_stor,
                'test1_err': test1_err_stor, 'test2_pred': test2_pred_stor, 'test2_err': test2_err_stor,
                'loss': loss_stor}, filename)
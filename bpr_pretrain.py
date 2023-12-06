import numpy as np
from utils import setup_seed
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import torch.optim as optim
import torch
import torch.nn as nn
import os
import time
from dataset import load_data
from sklearn.metrics import roc_auc_score
import argparse


class BPRData(data.Dataset):
    def __init__(self, x_train, num_user,
                 num_item, num_ng=0, is_training=None):
        """features=train_data,num_item=item_num
        """
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        train_data = x_train.tolist()
        train_mat = sp.dok_matrix((num_user, num_item), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

        self.features = train_data
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        # return -(prediction_i - prediction_j).sigmoid().log().sum()
        # return -(prediction_i - prediction_j).sigmoid().sum()
        return (prediction_j - prediction_i).sigmoid().sum()

    def fit(self, train_loader, lr, lamb, num_epoch, x_test, y_test):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        for epoch in range(1, num_epoch + 1):
            train_loader.dataset.ng_sample()
            train_loss = []
            for user, item_i, item_j in train_loader:
                # to cuda
                user = user.cuda()
                item_i = item_i.cuda()
                item_j = item_j.cuda()
                # train
                self.zero_grad()
                loss = self.forward(user, item_i, item_j)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            pred_test = self.predict(x_test)
            test_auc = roc_auc_score(y_test, pred_test)
            print(epoch, '\ttraining loss: ', np.mean(
                train_loss), '\ttest auc (ub):', test_auc)

    def predict(self, x_test):
        with torch.no_grad():
            user_id = torch.tensor(x_test[:, 0]).cuda()
            item_id = torch.tensor(x_test[:, 1]).cuda()
            user = self.embed_user(user_id)
            item = self.embed_item(item_id)
            pred = (user * item).sum(dim=-1)
        return pred.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='coat')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num_ng', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--factor_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lamb', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--save_path', type=str, default='pretrain')
    return parser.parse_args()

def save_pretrain(model: BPR, save_path, data_name):
    path = os.path.join(save_path, data_name, 'embedding.npz')
    user = model.embed_user.weight.detach().cpu().numpy()
    item = model.embed_item.weight.detach().cpu().numpy()
    np.savez(path, user=user, item=item)


def main():
    args = parse_args()
    print(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    x_train, y_train, x_test, y_test, num_user, num_item = load_data(
        args.data_name)
    train_dataset = BPRData(x_train, num_user, num_item,
                            num_ng=args.num_ng, is_training=True)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BPR(num_user, num_item, args.factor_num)
    model.cuda()
    model.fit(train_loader, args.lr, args.lamb, args.num_epoch, x_test, y_test)
    save_pretrain(model, args.save_path, args.data_name)


if __name__ == '__main__':
    main()

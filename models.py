# -*- coding: utf-8 -*-
from typing import DefaultDict
import pandas as pd
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import os


def get_model(args, num_user, num_item):
    path = os.path.join(args.pretrained_doc, args.data_name, args.pretrained_file)
    Model = eval(args.model_name)
    return Model(num_user, num_item, args.factor_num, pretrained_path=path)


def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i, j] for j in range(num_item)])
    return np.array(sample)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# ............

class NCF_CTW_1(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_CTW_1, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        # CTW parameters
        ## Treatment
        pretrained = np.load(kwargs['pretrained_path'])
        pre_user_embed = torch.tensor(pretrained['user']).cuda()
        pre_item_embed = torch.tensor(pretrained['item']).cuda()
        self.W_T = nn.Embedding.from_pretrained(pre_user_embed, freeze=True)
        self.H_T = nn.Embedding.from_pretrained(pre_item_embed, freeze=True)

        ## Confounder
        self.W_C = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H_C = torch.nn.Embedding(self.num_items, self.embedding_k)

        ## Instrumental
        self.W_I = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H_I = torch.nn.Embedding(self.num_items, self.embedding_k)

        ## Adjustment
        self.W_A = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H_A = torch.nn.Embedding(self.num_items, self.embedding_k)
        

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = 0)

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_z = self.W(user_idx)
        V_z = self.H(item_idx)

        U_T = self.W_T(user_idx)
        V_T = self.H_T(item_idx)

        U_I = self.W_I(user_idx)
        V_I = self.H_I(item_idx)

        U_C = self.W_C(user_idx)
        V_C = self.H_C(item_idx)

        U_A = self.W_A(user_idx)
        V_A = self.H_A(item_idx)

        z_emb = torch.cat([U_z, V_z], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        out = self.linear_2(h1)
        out = out + self.user_bias(user_idx) + self.item_bias(item_idx)

        a_emb = torch.cat([U_A, V_A], axis=1)
        h1_a = self.linear_1(a_emb)
        h1_a = self.relu(h1_a)
        out_a = self.linear_2(h1_a) + self.user_bias(user_idx) + self.item_bias(item_idx)

        t_emb = torch.cat([U_T, V_T], axis=1)
        i_emb = torch.cat([U_I, V_I], axis=1)
        h1_i = self.linear_1(i_emb)
        h1_i = self.relu(h1_i)
        out_i = self.linear_2(h1_i)

        u_div_reg_z = torch.norm((U_z - U_T / 2), p=2, dim=1) ** 2
        v_div_reg_z = torch.norm((V_z - V_T / 2), p=2, dim=1) ** 2
        div_reg_z = u_div_reg_z.mean() + v_div_reg_z.mean()

        u_div_reg_i = torch.norm((U_I - U_T / 2), p=2, dim=1) ** 2
        v_div_reg_i = torch.norm((V_T - V_T / 2), p=2, dim=1) ** 2
        div_reg_i = u_div_reg_i.mean() + v_div_reg_i.mean()

        u_div_reg_a = torch.norm((U_A - U_T / 2), p=2, dim=1) ** 2
        v_div_reg_a = torch.norm((V_A - V_T / 2), p=2, dim=1) ** 2
        div_reg_a = u_div_reg_a.mean() + v_div_reg_a.mean()


        if is_training:
            return out, out_a, out_i, div_reg_z, div_reg_i, div_reg_a
        else:
            return out

    def fit(self, args, x_train, y_train):
        self.alpha = args.alpha
        self.gamma = args.gamma

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)

        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                out, out_a, out_i, div_reg_z, div_reg_i, div_reg_a = self.forward(sub_x, True)
                pred = self.sigmoid(out)
                xent_loss = self.xent_func(torch.squeeze(pred), sub_y)

                pred_a = self.sigmoid(out_a)
                xent_loss_a = self.xent_func(torch.squeeze(pred_a), sub_y)

                pred_i = self.sigmoid(out_i)
                xent_loss_i = self.xent_func(torch.squeeze(pred_i), sub_y)

                loss = xent_loss + 0.0001 * xent_loss_a - 0.0001 * xent_loss_i + self.gamma * div_reg_z + 0.001 * div_reg_i - 0.001 * div_reg_a

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)

            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-CTW] Reach preset epochs, it seems does not converge.")

        def _compute_IPS(self, x, y, y_ips=None):
            if y_ips is None:
                one_over_zl = np.ones(len(y))
            else:
                py1 = y_ips.sum() / len(y_ips)
                py0 = 1 - py1
                po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
                py1o1 = y.sum() / len(y)
                py0o1 = 1 - py1o1

                propensity = np.zeros(len(y))

                propensity[y == 0] = (py0o1 * po1) / py0
                propensity[y == 1] = (py1o1 * po1) / py1
                one_over_zl = 1 / propensity

            one_over_zl = torch.Tensor(one_over_zl)
            return one_over_zl

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

# ............


class MF_ST(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_ST, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        # CTW parameters
        pretrained = np.load(kwargs['pretrained_path'])
        pre_user_embed = torch.tensor(pretrained['user']).cuda()
        pre_item_embed = torch.tensor(pretrained['item']).cuda()
        self.W_pre = nn.Embedding.from_pretrained(pre_user_embed, freeze=True)
        self.H_pre = nn.Embedding.from_pretrained(pre_item_embed, freeze=True)
        self.W_eps = nn.Parameter(torch.ones(num_users))
        self.H_eps = nn.Parameter(torch.ones(num_items))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        U_emb_pre = self.W_pre(user_idx)
        V_emb_pre = self.H_pre(item_idx)
        U_eps = self.W_eps[user_idx]
        V_eps = self.H_eps[item_idx]
        U_emb_b = U_emb + (U_eps ** 2).unsqueeze(-1) * U_emb_pre
        V_emb_b = V_emb + (V_eps ** 2).unsqueeze(-1) * V_emb_pre

        out = torch.sum(U_emb.mul(V_emb), 1)
        out_b = torch.sum(U_emb_b.mul(V_emb_b), 1)

        if is_training:
            return out, U_emb, V_emb, U_emb_pre, V_emb_pre, out_b
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        self.alpha_1 = args.alpha_1
        self.alpha_2 = args.alpha_2
        self.alpha_3 = args.alpha_3

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb, u_emb_pre, v_emb_pre, pred_b = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred, sub_y)

                pred_b = self.sigmoid(pred_b)
                xent_loss_b = self.xent_func(pred_b, sub_y)
                
                u_div_reg = torch.norm((u_emb - u_emb_pre), p=2, dim=1) ** 2
                v_div_reg = torch.norm((v_emb - v_emb_pre), p=2, dim=1) ** 2
                div_reg = u_div_reg.mean() + v_div_reg.mean()

                info_loss = self.alpha_3 * div_reg

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-ST] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-ST] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-ST] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

class MF_CTW(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CTW, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        # CTW parameters
        pretrained = np.load(kwargs['pretrained_path'])
        pre_user_embed = torch.tensor(pretrained['user']).cuda()
        pre_item_embed = torch.tensor(pretrained['item']).cuda()
        self.W_pre = nn.Embedding.from_pretrained(pre_user_embed, freeze=True)
        self.H_pre = nn.Embedding.from_pretrained(pre_item_embed, freeze=True)
        self.W_eps = nn.Parameter(torch.ones(num_users))
        self.H_eps = nn.Parameter(torch.ones(num_items))

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        U_emb_pre = self.W_pre(user_idx)
        V_emb_pre = self.H_pre(item_idx)
        U_eps = self.W_eps[user_idx]
        V_eps = self.H_eps[item_idx]
        U_emb_b = U_emb + (U_eps ** 2).unsqueeze(-1) * U_emb_pre
        V_emb_b = V_emb + (V_eps ** 2).unsqueeze(-1) * V_emb_pre

        out = torch.sum(U_emb.mul(V_emb), 1)
        out_b = torch.sum(U_emb_b.mul(V_emb_b), 1)

        if is_training:
            return out, U_emb, V_emb, U_emb_pre, V_emb_pre, out_b
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        self.alpha = args.alpha
        self.gamma = args.gamma

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb, u_emb_pre, v_emb_pre, pred_b = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred, sub_y)

                pred_b = self.sigmoid(pred_b)
                xent_loss_b = self.xent_func(pred_b, sub_y)
                
                u_div_reg = torch.norm((u_emb - u_emb_pre), p=2, dim=1) ** 2
                v_div_reg = torch.norm((v_emb - v_emb_pre), p=2, dim=1) ** 2
                div_reg = u_div_reg.mean() + v_div_reg.mean()

                info_loss = self.alpha * xent_loss_b + self.gamma * div_reg

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-CTW] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

class NCF_CTW(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_CTW, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

         # CTW parameters
        pretrained = np.load(kwargs['pretrained_path'])
        pre_user_embed = torch.tensor(pretrained['user']).cuda()
        pre_item_embed = torch.tensor(pretrained['item']).cuda()
        self.W_pre = nn.Embedding.from_pretrained(pre_user_embed, freeze=True)
        self.H_pre = nn.Embedding.from_pretrained(pre_item_embed, freeze=True)
        self.W_eps = nn.Parameter(torch.ones(num_users))
        self.H_eps = nn.Parameter(torch.ones(num_items))

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = 0)
        # nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = 0)

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        U_emb_pre = self.W_pre(user_idx)
        V_emb_pre = self.H_pre(item_idx)
        U_eps = self.W_eps[user_idx]
        V_eps = self.H_eps[item_idx]
        U_emb_b = U_emb + (U_eps ** 2).unsqueeze(-1) * U_emb_pre
        V_emb_b = V_emb + (V_eps ** 2).unsqueeze(-1) * V_emb_pre

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        out = self.linear_2(h1)
        out = out + self.user_bias(user_idx) + self.item_bias(item_idx)

        # with bias term
        z_emb_b = torch.cat([U_emb_b, V_emb_b], axis=1)
        h1_b = self.linear_1(z_emb_b)
        h1_b = self.relu(h1_b)
        out_b = self.linear_2(h1_b)
        out_b = out_b + self.user_bias(user_idx) + self.item_bias(item_idx)

        if is_training:
            return out, U_emb, V_emb, U_emb_pre, V_emb_pre, out_b
        else:
            return out

    def fit(self, args, x_train, y_train):
        self.alpha = args.alpha
        self.gamma = args.gamma

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)

        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb, u_emb_pre, v_emb_pre, pred_b = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(torch.squeeze(pred), sub_y)

                pred_b = self.sigmoid(pred_b)
                xent_loss_b = self.xent_func(pred_b, sub_y)
                
                u_div_reg = torch.norm((u_emb - u_emb_pre), p=2, dim=1) ** 2
                v_div_reg = torch.norm((v_emb - v_emb_pre), p=2, dim=1) ** 2
                div_reg = u_div_reg.mean() + v_div_reg.mean()

                info_loss = self.alpha * xent_loss_b + self.gamma * div_reg

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)

            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-CTW] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-CTW] Reach preset epochs, it seems does not converge.")

        def _compute_IPS(self, x, y, y_ips=None):
            if y_ips is None:
                one_over_zl = np.ones(len(y))
            else:
                py1 = y_ips.sum() / len(y_ips)
                py0 = 1 - py1
                po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
                py1o1 = y.sum() / len(y)
                py0o1 = 1 - py1o1

                propensity = np.zeros(len(y))

                propensity[y == 0] = (py0o1 * po1) / py0
                propensity[y == 1] = (py1o1 * po1) / py1
                one_over_zl = 1 / propensity

            one_over_zl = torch.Tensor(one_over_zl)
            return one_over_zl

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        self.alpha = args.alpha
        self.gamma = args.gamma

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x_train)
        batch_size = args.batch_size
        total_batch = num_sample // batch_size
        early_stop = 0

        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred, sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx * batch_size:(idx+1)*batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                logp_hat = pred.log()

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()

                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (
                    1-pred_ul_avg).log()) + self.gamma * torch.mean(pred * logp_hat)

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out


    def fit(self, args, x_train, y_train, x_test, y_test):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size

        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred, sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF] epoch:{}, xent:{}".format(epoch + 1, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

class MF_DIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_DIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.W_r = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H_r = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        U_emb_r = self.W_r(user_idx)
        V_emb_r = self.H_r(item_idx)

        U_emb_all = torch.cat([U_emb, U_emb_r], dim=-1)
        V_emb_all = torch.cat([V_emb, V_emb_r], dim=-1)

        out = torch.sum(U_emb.mul(V_emb), 1)
        out_r = torch.sum(U_emb_r.mul(V_emb_r), 1)
        out_all = torch.sum(U_emb_all.mul(V_emb_all), 1)

        if is_training:
            return out, out_r, out_all, U_emb, V_emb #U_emb_r, V_emb_r, U_emb_all, V_emb_all
        else:
            return out


    def fit(self, args, x_train, y_train, x_test, y_test):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size

        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, pred_r, pred_all, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred, sub_y)

                pred_r = self.sigmoid(pred_r)
                xent_loss_r = self.xent_func(pred_r, sub_y)

                pred_all = self.sigmoid(pred_all)
                xent_loss_all = self.xent_func(pred_all, sub_y)

                loss = (1 - args.gamma_d) * xent_loss \
                    - (args.gamma_d - args.alpha_d) * xent_loss_r \
                    + args.gamma_d * xent_loss_all

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-DIB] epoch:{}, xent:{}".format(epoch + 1, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-DIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_IPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        early_stop = 0
        one_over_zl = self._compute_IPS(x_train, y_train, y_ips=y_ips).cuda()

        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                                                   weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class MF_IMP(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_IMP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x_train, y_train)
        # else:
        one_over_zl = self._compute_IPS(x_train, y_train).cuda()

        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * args.batch_size:(idx+1)*args.batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(
                    pred, sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.Tensor([prior_y]*selected_idx.shape[0]).cuda()
                imputation_loss = F.binary_cross_entropy(
                    pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(
                    pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-IMP] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-IMP] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-IMP] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_DR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size

        # if y_ips is None:
        #     one_over_zl = self._compute_IPS(x_train, y_train)
        # else:
        one_over_zl = self._compute_IPS(x_train, y_train, y_ips).cuda()

        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * args.batch_size:(idx+1)*args.batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(
                    pred, sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.Tensor([prior_y]*selected_idx.shape[0]).cuda()
                imputation_loss = F.binary_cross_entropy(
                    pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(
                    pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_SNIPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        total_batch = num_sample // args.batch_size
        
        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]
        
        one_over_zl = self._compute_IPS(x_train, y_train, y_ips=y_ips).cuda()

        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[args.batch_size*idx:(idx+1)*args.batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                sum_inv_prop = torch.sum(inv_prop)

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                                                   weight=inv_prop, reduction="sum")

                xent_loss = xent_loss / sum_inv_prop

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy()

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


"""Neural Collaborative Filtering based methods.
"""


class NCF(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = self.xent_func(
                    pred, torch.unsqueeze(torch.Tensor(sub_y).cuda(), 1))

                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)


class NCF_IPS(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_IPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        batch_size = args.batch_size
        num_sample = len(x_train)
        total_batch = num_sample // batch_size

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        one_over_zl = self._compute_IPS(x_train, y_train, y_ips).cuda()

        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = torch.Tensor(y_train[selected_idx]).cuda()

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y,
                                                   weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[NCF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class NCF_SNIPS(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_SNIPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        batch_size = args.batch_size
        num_sample = len(x_train)
        total_batch = num_sample // batch_size

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        one_over_zl = self._compute_IPS(x_train, y_train, y_ips).cuda()

        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = torch.Tensor(y_train[selected_idx]).cuda()

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                sum_inv_prop = torch.sum(inv_prop)

                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                loss = F.binary_cross_entropy(torch.squeeze(pred), sub_y,
                                              weight=inv_prop, reduction="sum")
                loss = loss / sum_inv_prop

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[NCF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class NCF_CVIB(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        self.alpha = args.alpha
        self.gamma = args.gamma

        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x_train)
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        total_batch = num_sample // batch_size
        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = self.xent_func(torch.squeeze(pred), sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx * batch_size:(idx+1)*batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()
                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (
                    1-pred_ul_avg).log()) + self.gamma * torch.mean(pred * pred.log())

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)

            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)


class NCF_DR(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_DR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        batch_size = args.batch_size
        num_sample = len(x_train)
        total_batch = num_sample // batch_size

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        one_over_zl = self._compute_IPS(x_train, y_train, y_ips).cuda()

        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * batch_size:(idx+1)*batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(torch.squeeze(
                    pred), sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.unsqueeze(
                    torch.Tensor([prior_y]*selected_idx.shape[0]), 1).cuda()
                imputation_loss = F.binary_cross_entropy(
                    pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(
                    pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[NCF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl

class NCF_IMP(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_IMP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        batch_size = args.batch_size
        num_sample = len(x_train)
        total_batch = num_sample // batch_size

        ips_idxs = np.arange(len(y_test))
        np.random.shuffle(ips_idxs)
        y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

        one_over_zl = self._compute_IPS(x_train, y_train).cuda()

        prior_y = y_ips.mean()
        early_stop = 0
        for epoch in range(args.num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[idx * batch_size:(idx+1)*batch_size]]

                pred_ul, _, _ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(torch.squeeze(
                    pred), sub_y, weight=inv_prop, reduction="sum")

                imputation_y = torch.unsqueeze(
                    torch.Tensor([prior_y]*selected_idx.shape[0]), 1).cuda()
                imputation_loss = F.binary_cross_entropy(
                    pred, imputation_y, reduction="sum")

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]

                # direct loss
                direct_loss = F.binary_cross_entropy(
                    pred_ul, imputation_y, reduction="mean")

                loss = ips_loss + direct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-IMP] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-IMP] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == args.num_epoch - 1:
                print("[NCF-IMP] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)

    def _compute_IPS(self, x, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:, 0].max() * x[:, 1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl

class NCF_DIB(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_DIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.W_r = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H_r = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        U_emb_r = self.W_r(user_idx)
        V_emb_r = self.H_r(item_idx)

        U_emb_all = U_emb + U_emb_r
        V_emb_all = V_emb + V_emb_r

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        z_emb_r = torch.cat([U_emb_r, V_emb_r], axis=1)
        z_emb_all = torch.cat([U_emb_all, V_emb_all], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)
        out = self.linear_2(h1)

        h1_r = self.linear_1(z_emb_r)
        h1_r = self.relu(h1_r)
        out_r = self.linear_2(h1_r)

        h1_all = self.linear_1(z_emb_all)
        h1_all = self.relu(h1_all)
        out_all = self.linear_2(h1_all)

        # out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, out_r, out_all, U_emb, V_emb
        else:
            return out

    def fit(self, args, x_train, y_train, x_test, y_test):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.lamb)
        last_loss = 1e9

        num_sample = len(x_train)
        batch_size = args.batch_size
        num_epoch = args.num_epoch
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_train[selected_idx]
                sub_y = y_train[selected_idx]

                optimizer.zero_grad()
                pred, pred_r, pred_all, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(
                    pred, torch.unsqueeze(torch.Tensor(sub_y).cuda(), 1))

                pred_r = self.sigmoid(pred_r)
                xent_loss_r = self.xent_func(
                    pred_r, torch.unsqueeze(torch.Tensor(sub_y).cuda(), 1))
                
                pred_all = self.sigmoid(pred_all)
                xent_loss_all = self.xent_func(
                    pred_all, torch.unsqueeze(torch.Tensor(sub_y).cuda(), 1))

                # loss = xent_loss
                loss = (1 - args.gamma_d) * xent_loss \
                    - (args.gamma_d - args.alpha_d) * xent_loss_r \
                    + args.gamma_d * xent_loss_all
                
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.cpu().detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < args.tol:
                if early_stop > 5:
                    print("[NCF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0:
                print("[NCF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[NCF-DIB, Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.cpu().detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1-pred, pred], axis=1)



def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x, 1), torch.unsqueeze(x, 1)], axis=1)
    return out


def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)

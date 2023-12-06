import numpy as np
from utils import setup_seed, mse_func, ndcg_func, auc_func
from models import get_model
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import torch.optim as optim
import torch
import torch.nn as nn
import os
import time
from dataset import load_data
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Namespace(alpha=10, 
    # batch_size=128, 
    # data_name='coat', 
    # factor_num=4, 
    # gamma=1e-05, 
    # gpu_id='7', 
    # lamb=0.0001, 
    # lr=0.01, model_name='MF', 
    # num_epoch=1000, 
    # pretrained_doc='pretrain', 
    # pretrained_file='embedding.npz', 
    # seed=2020, tol=1e-05)

    parser.add_argument('--data_name', type=str, default='coat')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--factor_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lamb', type=float, default=1e-05)
    parser.add_argument('--gpu_id', type=str, default='7')

    parser.add_argument('--model_name', type=str, default='MF')
    parser.add_argument('--tol', type=float, default=1e-5)

    # CVIB
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0.05)

    # DIB
    parser.add_argument('--alpha_d', type=float, default=0.01)
    parser.add_argument('--gamma_d', type=float, default=0.1)

    # CTW
    parser.add_argument('--alpha_1', type=float, default=0.001)
    parser.add_argument('--alpha_2', type=float, default=0.0)
    parser.add_argument('--alpha_3', type=float, default=1.0)
    parser.add_argument('--pretrained_doc', type=str, default='pretrain')
    parser.add_argument('--pretrained_file', type=str, default='embedding.npz')
    return parser.parse_args()


def evaluate(model, x_test, y_test):
    # Performance per user
    mse_res = mse_func(model, x_test, y_test)
    auc_res = auc_func(model, x_test, y_test)
    ndcg_res = ndcg_func(model, x_test, y_test)

    # Avgerage performance over all users
    mse = np.mean(mse_res)
    auc = np.mean(auc_res)
    ndcg_5 = np.mean(ndcg_res["ndcg_5"])
    ndcg_10 = np.mean(ndcg_res["ndcg_10"])

    print("***"*5 + "[####]" + "***"*5)
    print("test mse:", mse)
    print("test auc:", auc)
    print("ndcg_5:", ndcg_5)
    print("ndcg_10:", ndcg_10)


def main():
    args = parse_args()
    print(args)

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    x_train, y_train, x_test, y_test, num_user, num_item = load_data(
        args.data_name)
    model = get_model(args, num_user, num_item)
    model.cuda()
    model.fit(args, x_train, y_train, x_test, y_test)
    evaluate(model, x_test, y_test)


if __name__ == '__main__':
    main()

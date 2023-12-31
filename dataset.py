# -*- coding: utf-8 -*-

import numpy as np
import os
from utils import shuffle, binarize


data_dir = "./data"


def load_data(dataset_name="coat"):
    if dataset_name == "coat":
        train_mat, test_mat = _load_data("coat")        
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]
    elif dataset_name == "yahoo":
        x_train, y_train, x_test, y_test = _load_data("yahoo")
        x_train, y_train = shuffle(x_train, y_train)
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
    y_train = binarize(y_train)
    y_test = binarize(y_test)
    return x_train, y_train, x_test, y_test, num_user, num_item
    

def _load_data(name="coat"):

    if name == "coat":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir, "train.ascii")
        test_file = os.path.join(data_set_dir, "test.ascii")

        with open(train_file, "r") as f:
            x_train = []
            for line in f.readlines():
                x_train.append(line.split())

            x_train = np.array(x_train).astype(int)

        with open(test_file, "r") as f:
            x_test = []
            for line in f.readlines():
                x_test.append(line.split())

            x_test = np.array(x_test).astype(int)

        print("===>Load from {} data set<===".format(name))
        print("[train] rating ratio: {:.6f}".format(
            (x_train > 0).sum() / (x_train.shape[0] * x_train.shape[1])))
        print("[test]  rating ratio: {:.6f}".format(
            (x_test > 0).sum() / (x_test.shape[0] * x_test.shape[1])))

    elif name == "yahoo":
        data_set_dir = os.path.join(data_dir, name)
        train_file = os.path.join(data_set_dir,
                                  "ydata-ymusic-rating-study-v1_0-train.txt")
        test_file = os.path.join(data_set_dir,
                                 "ydata-ymusic-rating-study-v1_0-test.txt")

        x_train = []
        # <user_id> <song id> <rating>
        with open(train_file, "r") as f:
            for line in f:
                x_train.append(line.strip().split())
        x_train = np.array(x_train).astype(int)

        x_test = []
        # <user_id> <song id> <rating>
        with open(test_file, "r") as f:
            for line in f:
                x_test.append(line.strip().split())
        x_test = np.array(x_test).astype(int)
        print("===>Load from {} data set<===".format(name))
        print("[train] num data:", x_train.shape[0])
        print("[test]  num data:", x_test.shape[0])

        return x_train[:, :-1], x_train[:, -1], \
            x_test[:, :-1], x_test[:, -1]

    else:
        print("Cant find the data set", name)
        return

    return x_train, x_test


def rating_mat_to_sample(mat):
    row, col = np.nonzero(mat)
    y = mat[row, col]
    x = np.concatenate([row.reshape(-1, 1), col.reshape(-1, 1)], axis=1)
    return x, y

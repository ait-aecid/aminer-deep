import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


# https://blog.csdn.net/folk_/article/details/80208557
def train_val_split(logs_meta, labels, val_ratio=0.1):
    total_num = len(labels)
    train_index = list(range(total_num))
    train_logs = {}
    val_logs = {}
    for key in logs_meta.keys():
        train_logs[key] = []
        val_logs[key] = []
    train_labels = []
    val_labels = []
    val_num = int(total_num * val_ratio)

    for i in range(val_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        for key in logs_meta.keys():
            val_logs[key].append(logs_meta[key][random_index])
        val_labels.append(labels[random_index])
        del train_index[random_index]

    for i in range(total_num - val_num):
        for key in logs_meta.keys():
            train_logs[key].append(logs_meta[key][train_index[i]])
        train_labels.append(labels[train_index[i]])

    return train_logs, train_labels, val_logs, val_labels

def load_sequences(data_dir):
    """
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    """

    num_seq = 0
    result_logs = {}
    result_logs["Sequentials"] = []
    groups = []
    labels = []

    with open(data_dir, "r") as f:
        for line in f.readlines():
            group, seq, label, result = line.strip(":").split(":")
            seq = seq.replace("[", "").replace("]", "").replace(",", "")
            # seq=list(map(lambda n: n, map(int, seq.strip().split())))
            group, seq, label = (
                int(group),
                list(map(lambda n: n, map(int, seq.strip().split()))),
                int(label),
            )
            Sequential_pattern = seq
            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            result_logs["Sequentials"].append(Sequential_pattern)
            labels.append(label)
            groups.append(group)
            num_seq += 1
        print("File {}, number of sequences {}".format(data_dir, num_seq))
    return result_logs, labels, groups

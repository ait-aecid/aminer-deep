"""
Description : This file implements the sliding window
Author      : https://github.com/donglee-afar
License     : MIT
"""

from collections import Counter
import numpy as np
import pandas as pd

def data_sp_sampling(logs, lables, groups, result, sample_ratio):
    train_num_seq = 0
    train_logs = {}
    train_logs["Sequentials"] = []
    train_groups = []
    train_labels = []
    train_result = []

    test_num_seq = 0
    test_logs = {}
    test_logs["Sequentials"] = []
    test_groups = []
    test_labels = []
    test_result = []

    group_nd_set = pd.DataFrame(set(groups))
    train_sample_group = group_nd_set.sample(frac=(1 - sample_ratio), random_state=200)
    # test_sample_group = group_nd_set.drop(train_sample_group.index)

    for key in range(len(lables)):
        if groups[key] in train_sample_group.values:
            train_num_seq += 1
            train_logs["Sequentials"].append(logs["Sequentials"][key])
            train_labels.append(lables[key])
            train_groups.append(groups[key])
            train_result.append(result[key])
        else:
            test_num_seq += 1
            test_logs["Sequentials"].append(logs["Sequentials"][key])
            test_labels.append(lables[key])
            test_groups.append(groups[key])
            test_result.append(result[key])
    print(
        "Number of tranining sequences {}, and test sequences {}".format(
            train_num_seq, test_num_seq
        )
    )
    return train_logs, train_labels, train_groups, train_result, test_logs, test_labels, test_groups, test_result


def data_sampling(logs, lables, groups, sample_ratio):
    train_num_seq = 0
    train_logs = {}
    train_logs["Sequentials"] = []
    train_groups = []
    train_labels = []

    test_num_seq = 0
    test_logs = {}
    test_logs["Sequentials"] = []
    test_groups = []
    test_labels = []

    group_nd_set = pd.DataFrame(set(groups))
    train_sample_group = group_nd_set.sample(frac=(1 - sample_ratio), random_state=200)
    # test_sample_group = group_nd_set.drop(train_sample_group.index)

    for key in range(len(lables)):
        if groups[key] in train_sample_group.values:
            train_num_seq += 1
            train_logs["Sequentials"].append(logs["Sequentials"][key])
            train_labels.append(lables[key])
            train_groups.append(groups[key])
        else:
            test_num_seq += 1
            test_logs["Sequentials"].append(logs["Sequentials"][key])
            test_labels.append(lables[key])
            test_groups.append(groups[key])
    print(
        "Number of tranining sequences {}, and test sequences {}".format(
            train_num_seq, test_num_seq
        )
    )
    return train_logs, train_labels, train_groups, test_logs, test_labels, test_groups


def load_raw_data(data_dir):
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
    results = []

    with open(data_dir, "r") as f:
        for line in f.readlines():
            group, seq, label, result = line.strip(":").split(":")
            seq = seq.replace("[", "").replace("]", "").replace(",", "")
            group, seq, label, result = (
                group,
                list(map(lambda n: n, map(int, seq.strip().split()))),
                int(label), 
                bool(result),
            )
            Sequential_pattern = seq
            # Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            result_logs["Sequentials"].append(Sequential_pattern)
            labels.append(label)
            groups.append(group)
            results.append(result)
            num_seq += 1
        print("File {}, number of sequences {}".format(data_dir, num_seq))
    return result_logs, labels, groups, results

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

    with open(data_dir, "r") as f:
        for line in f.readlines():
            group, seq, label, result = line.strip(":").split(":")
            seq = seq.replace("[", "").replace("]", "").replace(",", "")
            group, seq = (
                group,
                list(map(lambda n: n, map(int, seq.strip().split()))),
            )
            Sequential_pattern = seq
            # Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            result_logs["Sequentials"].append(Sequential_pattern)
            groups.append(group)
            num_seq += 1
        print("File {}, number of sequences {}".format(data_dir, num_seq))
    return result_logs, groups

def load_data(data_dir):
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
            group, seq, label = (
                group,
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

def sliding_window(data_dir, datatype, window_size, num_classes, padding_option, sample_ratio=1,):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''

    num_seq = 0
    result_logs = {}
    result_logs['Sequentials'] = []

    labels = []
    if datatype == 'train':
        data_dir = data_dir + '_train'
    if datatype == 'val':
        data_dir = data_dir + '_test_normal'

    with open(data_dir, 'r') as f:
        for line in f.readlines():

            line = list(map(lambda n: n, map(int, line.strip(":").split())))
            #Padding
            if padding_option:
                line = line + [17] * (window_size + 1 - len(line))

            if len(line) >= window_size:
                num_seq += 1
            
            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * num_classes
                log_counter = Counter(Sequential_pattern)
                
                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                   
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                labels.append(line[i + window_size])
            


    print('File {}, number of sequences {}'.format(data_dir, num_seq))
    return result_logs, labels
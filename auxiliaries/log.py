"""
Description : This file implements the log dataset class
Author      : https://github.com/donglee-afar
License     : MIT
"""

import torch
from torch.utils.data import Dataset


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True):
        self.seq = seq
        if self.seq:
            self.Sequentials = logs['Sequentials']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        return log, self.labels[idx]




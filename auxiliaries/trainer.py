"""
Description : This file implements the Trainer class
Author      : https://github.com/donglee-afar
License     : MIT
"""

import gc
import os
import time


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from auxiliaries.log import log_dataset
from auxiliaries.sample import *
from auxiliaries.utils import save_parameters
from tqdm import tqdm


class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        #self.data_dir = options['data_save_dir']
        #self.data_file = options['data_file_name']
        self.batch_size = options['batch_size']
        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']
        self.sequentials = options['sequentials']
        self.num_classes = options['num_classes']

        #os.makedirs(self.save_dir, exist_ok=True)
        #result, lables, groups = load_data(self.data_dir +  self.data_file)
        #train_logs, train_labels, train_groups, test_logs, test_labels, test_groups = data_sampling(result, lables, groups, 0.2)
    
        #train_dataset = log_dataset(logs=train_logs,
        #                            labels=train_labels,
        #                            seq=self.sequentials)
        #valid_dataset = log_dataset(logs=test_logs,
        #                            labels=test_labels,
        #                            seq=self.sequentials)

        #del result, lables, groups
        #del train_logs, train_labels, train_groups, test_logs, test_labels, test_groups
        #gc.collect()

        #self.train_loader = DataLoader(train_dataset,
        #                               batch_size=self.batch_size,
        #                               shuffle=True,
        #                               pin_memory=True)
        #self.valid_loader = DataLoader(valid_dataset,
        #                               batch_size=self.batch_size,
        #                               shuffle=False,
        #                               pin_memory=True)


        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch, data, labels):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        #Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        train_dataset = log_dataset(logs=data,
                                    labels=labels,
                                    seq=self.sequentials)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
        tbar = tqdm(train_loader, desc="\r")
        num_batch = len(train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            #print(str(log) + ' -> ' + str(label))
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            #Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                #Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
                self.optimizer.step()
                #before next iteration we must call zero_grad() -> empties the gradients
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch, data, labels):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        valid_dataset = log_dataset(logs=data,
                                    labels=labels,
                                    seq=self.sequentials)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
        tbar = tqdm(valid_loader, desc="\r")
        num_batch = len(valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")

    def start_train(self, data, labels):
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch, data, labels)
            #self.valid(epoch)
            #hdfs             if epoch >= self.max_epoch // 2 and epoch % 10 == 0:
            #if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
            #    self.valid(epoch, data, labels)
            #    self.save_checkpoint(epoch,
            #                         save_optimizer=True,
            #                         suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()


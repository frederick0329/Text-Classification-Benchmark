# -*- coding: utf-8 -*-
from __future__ import print_function
import glob
import unicodedata
import string
import torch
import torch.nn as nn
import time
import numpy as np
import re
import pickle
import argparse
from torch import optim
from torch.autograd import Variable
from GRU import GRU
from DataLoader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

class RecurrentNet:
    def __init__(self,index2word, emb_size, class_inv_freq, hidden_size, bidirectional, learning_rate, model_save_path, pretrained_path=""):
        self.softmax = nn.Softmax()
        self.model = GRU(index2word, emb_size, hidden_size, len(class_inv_freq), bidirectional, pretrained_path=pretrained_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_inv_freq))
        self.model_save_path = model_save_path

    def fit(self, train_data, dev_data, num_epochs):
        self.trainEpoch(train_data, dev_data, num_epochs)

    def predict(self, test_data):
        self.model = torch.load(self.model_save_path)
        return self.evaluate(test_data, test=True)    


    def train(self, train_data):
        self.model.train()
        iter_loss = 0
        for b in range(train_data.getNumBatches()):
            batch_size = train_data.getBatch(b)[0].shape[0]
            self.optimizer.zero_grad()
            x_b = torch.from_numpy(train_data.getBatch(b)[0]).long()
            y_b = torch.from_numpy(train_data.getBatch(b)[1][:,0]).long() 
      
            x_var = Variable(x_b)
            y_var = Variable(y_b)

            h = self.model.init_hidden(batch_size)

            y_prob, h = self.model(x_var, h)

            loss = self.criterion(y_prob, y_var)
            loss.backward()


            self.optimizer.step()
            iter_loss += loss.data[0]
        return iter_loss

    def evaluate(self, data, test=False):
        self.model.eval()    
        y_pred = torch.zeros(data.num_instances, 1)
        y_test = np.zeros((data.num_instances, 1))
        count = 0
        for b in range(data.getNumBatches()):
            batch_size = data.getBatch(b)[0].shape[0]
            x_b = torch.from_numpy(data.getBatch(b)[0]).long()
            y_b = torch.from_numpy(data.getBatch(b)[1][:,0]).long()

            x_var = Variable(x_b)
            y_var = Variable(y_b)

            h = self.model.init_hidden(batch_size)
            out, h = self.model(x_var, h)    
            y_prob = self.softmax(out)
            _, preds = torch.max(y_prob.data, dim=1)
            y_test[count:count+batch_size, :] = data.getBatch(b)[1]
            y_pred[count:count+batch_size, :] = preds
            count = count + batch_size 
        if test:
            return y_test, y_pred.numpy()
        else:
            return f1_score(y_test, y_pred.numpy(), average='binary')


    def trainEpoch(self, train_data, dev_data, num_epochs):
        best_f1 = 0.0
        best_epoch = None
        for epoch in range(num_epochs):
            print('Epoch {}'.format(epoch), end=', ')
            train_data.shuffleBatches()
            iter_loss = self.train(train_data)
            print('training Loss: {:.3}'.format(iter_loss / 1.0), end=', ')
            train_f1 = self.evaluate(train_data)
            print('train f1 score: {}'.format(train_f1), end=', ')
            dev_f1 = self.evaluate(dev_data)
            print('dev f1 score: {}'.format(dev_f1))
            if dev_f1 >= best_f1:
                best_f1 = dev_f1
                best_epoch = epoch
                torch.save(self.model, self.model_save_path)   
        print("Saved the best model. (epoch " + str(best_epoch) + ")")









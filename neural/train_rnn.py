#!/usr/bin/env python
from __future__ import print_function
import os
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
from RecurrentNet import RecurrentNet
from DataLoader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='RNN(GRU) classifier')
parser.add_argument('-max_length', type=int, default=60, help='max length of training data [default: 50]')
parser.add_argument('-min_length', type=int, default=2, help='min length of training data [default: 2]')
parser.add_argument('-max_batch_size', type=int, default=50, help='max size of batch [default: 50]')
parser.add_argument('-num_epochs', type=int, default=10, help='num of epochs [default: 50]')
parser.add_argument('-freq_threshold', type=int, default=1, help='tokens with less than this number will be set to <unk> [default: 5]')
parser.add_argument('-hidden_size', type=int, default=128, help='default hidden size [default: 64]')
parser.add_argument('-emb_size', type=int, default=300, help='size of word embedding [default: 128]')
parser.add_argument('-learning_rate', type=float, default=0.001, help='initial learning rate for adam [default: 0.001]')
parser.add_argument('-bidirectional', type=bool, default=True, help='using bidirectional RNN [default: True]')
parser.add_argument('-data_folder', type=str, default='./data/segment_level', help='data folder [default: ./data]')
parser.add_argument('-model_save_folder', type=str, default='./nerual/models/segment_level', help='model save folder folder [default: ./neural/models/segment_level]')
parser.add_argument('-labels_file', type=str, default='./data/labels.txt', help='labels file [default: ../data/labels.txt]')
parser.add_argument('-level', type=str, default='segment_level', help='labels file [default: ../data/labels.txt]')
#parser.add_argument('-pretrained_path', type=str, default='./GoogleNews-vectors-negative300.bin', help='pretrained word vectors path [default: ""]')
parser.add_argument('-pretrained_path', type=str, default='', help='pretrained word vectors path [default: ""]')
args = parser.parse_args()

def main():
    with open(args.labels_file, 'r') as f:
        labels = f.read().splitlines()
    '''
    model_folder = './models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    level_folder = model_folder + '/' + args.level
    if not os.path.exists(level_folder):
        os.makedirs(level_folder)
    '''
    for label in labels:
        category_folder = args.model_save_folder + '/' + label
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
    
    y_true_all = []
    y_pred_all = []
    for label in labels:
        print(label)
        train_folder = args.data_folder + '/' + label + '/train'
        test_folder = args.data_folder + '/' + label + '/test'
        model_save_path = args.model_save_folder + '/' + label + '/' + 'rnn' + '.model'
        dict_save_path = args.model_save_folder + '/' + label + '/' + 'rnn' + '.dict'
        dl = DataLoader(train_folder, test_folder, args.freq_threshold, args.max_batch_size, args.max_length, args.min_length)
        en_dict, train_data, dev_data, test_data = dl.prepareData()
        with open(dict_save_path, 'wb') as handle:
            pickle.dump(en_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        clf = RecurrentNet(en_dict.index2word, args.emb_size, dl.class_inv_freq, args.hidden_size, args.bidirectional, args.learning_rate, model_save_path, pretrained_path=args.pretrained_path)
        clf.fit(train_data, dev_data, args.num_epochs)
        y_true, y_pred = clf.predict(test_data)
        y_true_all.append(y_true[:,0])
        y_pred_all.append(y_pred[:,0])
    print(classification_report(np.array(y_true_all).T, np.array(y_pred_all).T, target_names=labels))

if __name__ == "__main__":
    main()

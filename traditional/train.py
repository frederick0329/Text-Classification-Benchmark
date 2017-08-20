#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import numpy as np

from tModels import tModels
from DataLoader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Traditional classifiers')
parser.add_argument('-num_folds', type=int, default=5, help='num of folds for cross validation [default: 5]')
parser.add_argument('-feature', type=str, default='tfidf', choices=['tfidf', 'tf'], help='features [default: tfidf]')
parser.add_argument('-classifier', type=str, default='lr', choices=['lr', 'svm', 'nbsvm', 'mnb', 'rf'], help='classfiers [default: lr]')

parser.add_argument('-train_file', type=str, default='/home/chieh/ff2/text_classification/data/First_Party_Collection_Use_train.txt', help='train file [default: ./data]')
parser.add_argument('-dev_file', type=str, default='', help='dev file [default: ./data]')
parser.add_argument('-test_file', type=str, default='/home/chieh/ff2/text_classification/data/First_Party_Collection_Use_test.txt', help='test file [default: ./data]')

parser.add_argument('-model_save_folder', type=str, default='./models', help='model save folder folder [default: ./neural/models/segment_level]')

args = parser.parse_args()

def main():
    model_save_folder = args.model_save_folder
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_save_path = args.model_save_folder + '/' + args.classifier + '.model'
    
    dl = DataLoader(args.train_file, args.test_file, args.num_folds)
    train_data, test_data = dl.prepareData(args.feature)
    clf = tModels(args.classifier, args.num_folds, args.feature, dl, model_save_path)
    clf.fit(train_data)
    y_true, y_pred = clf.predict(test_data)
    corrects = (y_true == y_pred).sum()
    print('testing accuracy: {:}'.format(corrects * 1.0 / y_true.shape[0]))

if __name__ == "__main__":
    main()

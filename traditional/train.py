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
parser.add_argument('-learning_rate', type=float, default=0.001, help='initial learning rate for adam [default: 0.001]')
parser.add_argument('-data_folder', type=str, default='./data/sentence_level', help='data folder [default: ./data]')
parser.add_argument('-model_save_folder', type=str, default='./traditional/models/sentence_level', help='model save folder folder [default: ./traditional/models/segment_level]')
parser.add_argument('-labels_file', type=str, default='./data/labels.txt', help='labels file [default: ../data/labels.txt]')
parser.add_argument('-level', type=str, default='sentence_level', help='labels file [default: ../data/labels.txt]')
args = parser.parse_args()

def main():
    with open(args.labels_file, 'r') as f:
        labels = f.read().splitlines()
    '''
    model_folder = args.model_save_folder
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
        model_save_path = args.model_save_folder + '/' + label + '/' + args.classifier + '.model'
        num_categories = 2
        dl = DataLoader(train_folder, test_folder, args.labels_file, args.num_folds)
        train_data, test_data = dl.prepareData(args.feature)
        clf = tModels(args.classifier, args.num_folds, args.feature, dl, model_save_path)
        clf.fit(train_data)
        y_true, y_pred = clf.predict(test_data)
        y_true_all.append(np.squeeze(y_true))
        y_pred_all.append(y_pred)
    print(classification_report(np.array(y_true_all).T, np.array(y_pred_all).T, target_names=labels))

if __name__ == "__main__":
    main()

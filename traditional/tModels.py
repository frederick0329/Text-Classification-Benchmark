#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from DataLoader import DataLoader
from NBSVM import NBSVM


class tModels():
    def __init__(self, classifier, num_folds, feature, dl, model_save_path):
        self.classifier = classifier
        self.model_save_path = model_save_path
        self.num_folds = num_folds
        self.feature = feature
        self.dl = dl
  
    def createClassifier(self, config):
        if self.classifier == "lr":
            return LogisticRegression(class_weight='balanced', penalty=config["penalty"], C=config["C"])
        elif self.classifier == "gnb":
            return GaussianNB()
        elif self.classifier == "gp":
            return GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
        elif self.classifier == "mnb":
            return MNB(alpha=config["alpha"], fit_prior=config["fit_prior"])
        elif self.classifier == "svm":
            return SVC(C=config["C"], kernel=config["kernel"], class_weight='balanced')
        elif self.classifier == "rf":
            return RFC(n_estimators=config["n_estimators"], class_weight='balanced')
        elif self.classifier == "dt":
            return DTC(criterion=config["criterion"], class_weight='balanced')
        elif self.classifier == "nbsvm":
            return NBSVM(C=config["C"], beta=config["beta"])

    def fit(self, train_data):
        self.best_config = self.crossValidation(self.num_folds, self.generateGrid(self.getParams(self.classifier)), self.classifier, self.feature, self.dl)
        self.clf = self.createClassifier(self.best_config)
        self.clf.fit(train_data[0], np.squeeze(train_data[1]))
        model = {}
        model['vectorizer'] = self.dl.vectorizer
        model['model'] = self.clf
        with open(self.model_save_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, test_data):
        return test_data[1], self.clf.predict(test_data[0])
   
    def evaluate(self, y_pred, y_true, test=False, labels=None):
        if test:
            print(classification_report(y_true, y_pred.T, target_names=labels)) 
        else:    
            #print(f1_score(y_true, y_pred.T, average='micro'))
            return f1_score(y_true, y_pred.T, average='binary')


    def crossValidation(self, folds, grid, classifier, feature, dl):
        max_f1 = 0
        for config in grid:
            micro_f1_sum = 0
            for i in range(folds):
                train_data, dev_data = dl.prepareData(feature, fold_id=i)
                clf = self.createClassifier(config)
                clf.fit(train_data[0], np.squeeze(train_data[1]))
                pred_y = clf.predict(dev_data[0])
                micro_f1_sum += self.evaluate(pred_y, dev_data[1], test=False)
            micro_f1_avg = float(micro_f1_sum) / float(folds)
            print('{}'.format(config), end=', ')
            print(' : {}'.format(micro_f1_avg))
            if micro_f1_avg >= max_f1:
                best_config = config
                max_f1 = micro_f1_avg
        print('best config: {}'.format(best_config))
        return best_config


    def generateGrid(self, params):
        p = []
        grid = []
        for param in params:
            p.append(param)
        self.helper(grid, p, 0, params, {})
        return grid

    def helper(self, grid, p, level, params, cur):
        if level == len(p):
            grid.append(cur.copy())
            return
        for value in params[p[level]]:
            cur[p[level]] = value
            self.helper(grid, p, level + 1, params, cur)
            cur.pop(p[level])
        

    def getParams(self, classfier):
        if classfier == "lr":
            return {"penalty": ["l1", "l2"], "C":[0.1, 1, 10]}
        elif classfier == "svm":
            return {"C":[0.1, 1, 10], "kernel":["linear", "poly", "rbf"]}
        elif classfier == "mnb":
            return {"alpha":[0.1, 0.5, 1, 2], "fit_prior":[True, False]} 
        elif classfier == "rf":
            return {"n_estimators":[10, 100]}
        elif classfier == "dt":
            return {"criterion":["gini", "entropy"]}
        elif classfier == 'nbsvm':
            return {"C":[0.1, 1, 10], "beta":[0.25, 0.5, 0.75, 1.0]}
        return {}


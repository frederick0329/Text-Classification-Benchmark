from __future__ import print_function
import csv
import re

def preprocess_ag_news():
    raw_train_file = './raw/ag_news_csv/train.csv' 
    raw_test_file = './raw/ag_news_csv/test.csv'
    train_file = './preprocessed/ag_news_train.txt'
    test_file = './preprocessed/ag_news_test.txt'
    f_train =  open(train_file,'w')
    f_test =  open(test_file,'w')


    with open(raw_train_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            txt = ""
            for s in row[1:]:
                txt = txt + " " + s.replace("\\", " ")
            f_train.write(str(int(row[0]) - 1) + '\t' + txt + '\n')
    f_train.close()
    
    with open(raw_test_file, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            txt = ""
            for s in row[1:]:
                txt = txt + " " + s.replace("\\", " ")
            f_test.write(str(int(row[0]) - 1) + '\t' + txt + '\n')
    f_test.close()


preprocess_ag_news()

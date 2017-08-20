import numpy as np
import glob
import re
import random
from Dict import Dict
from Data import Data
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
class DataLoader:
    def __init__(self, train_folder, test_folder, freq_threshold, max_batch_size, max_length, min_length):
        self.train_filenames = glob.glob(train_folder + '/*')
        self.test_filenames = glob.glob(test_folder + '/*')
        self.dev_ratio = 0.1
        self.en_dict = None
        self.MAX_LENGTH = max_length
        self.MIN_LENGTH = min_length
        self.max_batch_size = max_batch_size
        self.freq_threshold = freq_threshold
        self.class_inv_freq = None
    def indexesFromSentence(self, sent):        
        words = sent.split(' ')
        index = []
        for word in words:
            if word not in self.en_dict.word2index:
                index.append(self.en_dict.word2index['<unk>']) 
            else:
                index.append(self.en_dict.word2index[word]) 
        return index

    def boolsFromLabel(self, label):
        return [int(i) for i in label.strip().split(' ')]

    def normalizeString(self,string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " ( ", string) 
        string = re.sub(r"\)", " ) ", string) 
        string = re.sub(r"\?", " ? ", string) 
        string = re.sub(r"\s{2,}", " ", string)   
        return string.strip().lower()

    def lemmatizeString(self, s):
        return " ".join([wnl.lemmatize(i) for i in s.split()])

    def readInput(self, filenames):
        lines = []
        for filename in filenames:
            lines += open(filename, 'r').readlines()
    
        pairs = []
        for line in lines:
            label, sent = line.split('\t')
            sent = self.normalizeString(sent)
            pairs.append((sent, label))
        return pairs


    def trimSent(self, pair):
        tmp = pair[0].split()
        if len(tmp) <= self.MAX_LENGTH:
            return pair
        return (" ".join(tmp[:self.MAX_LENGTH]), pair[1])
    

    def trimSents(self, pairs):
        return [self.trimSent(pair) for pair in pairs]

    def countClassInvFreq(self, pairs):
        class_count = []
        for pair in pairs:
            while int(pair[1]) >= len(class_count):
                class_count.append(0)
            class_count[int(pair[1])] += 1
        self.class_inv_freq = [1.0 * len(pairs) / (len(class_count) * count) for count in class_count]

    def prepareData(self):
        num_train = int(round(len(self.train_filenames) * (1 - self.dev_ratio)))
        
        #random.shuffle(self.train_filenames)

        #read train data
        print("Reading Training Data...")
        pairs = self.readInput(self.train_filenames)
        train_pairs = self.readInput(self.train_filenames[:num_train])
        print("Read %s sentence pairs" % len(train_pairs))
        self.countClassInvFreq(train_pairs)
        train_pairs = self.trimSents(train_pairs)
        self.en_dict = Dict()
        for pair in train_pairs:
            self.en_dict.addSentence(pair[0])
  
        print("Reading Development Data...")
        dev_pairs = self.readInput(self.train_filenames[num_train:])
        print("Read %s sentence pairs" % len(dev_pairs))
        #pairs = self.trimSents(pairs)
        #for pair in dev_pairs:
        #    self.en_dict.addSentence(pair[0])

        print("Reading Testing Data...")
        test_pairs = self.readInput(self.test_filenames)
        print("Read %s sentence pairs" % len(test_pairs))
        #pairs = self.trimSents(pairs)
        #for pair in test_pairs:
        #    self.en_dict.addSentence(pair[0])

        print("Number of words before threshold: %d" % self.en_dict.n_words)
        self.en_dict.removeLowFreqWords(self.freq_threshold) 
        print("Number of words after threshold: %d" % self.en_dict.n_words)

        train = Data(self.stringToIndex(train_pairs), self.max_batch_size)
        dev = Data(self.stringToIndex(dev_pairs), self.max_batch_size)
        test = Data(self.stringToIndex(test_pairs), self.max_batch_size)

        

        #pairs = self.trimSents(pairs)
        return self.en_dict, train, dev, test

    def stringToIndex(self, pairs):
        ret = []
        for pair in pairs:
            ret.append((self.indexesFromSentence(pair[0]), self.boolsFromLabel(pair[1])))
        return ret 



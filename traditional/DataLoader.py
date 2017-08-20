import numpy as np
import glob
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_selection import SelectKBest, chi2
from scipy import sparse
wnl = WordNetLemmatizer()


class DataLoader:
    """
    Reads the input data and transofrm raw text and string labels into 
    numpy arrays.
    It also supports extracting devlopment set from training data with
    the fold_id parameter in the prepareData method for cross validation. 
    """
    def __init__(self, train_filename, test_filename, num_folds):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.num_folds = num_folds        
        self.vectorizer = None

    def readInput(self, filename):
        """
        Reads the input and extracts the text, labels, and segment index
        and pack them into a tuple.
        """
        x = []
        y = []
        pairs = []
        # Read the file and split into lines
        lines = open(filename, 'r').readlines()
        for i, line in enumerate(lines):
            label, sent = line.split('\t')
            sent = self.normalizeString(sent)
            x.append(sent)
            y.append(label)
        return (x, y)
    
    def normalizeString(self, string):
        """
        String tokenization
        """
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

    def stringToNumpy(self, pairs, vectorizer):
        """
        Transform raw text and string labels into numpy array with the vectorizer.
        """
        x = vectorizer.transform(pairs[0])
        y = np.array([[int(i) for i in label.strip().split(' ')] for label in pairs[1]])
        return (x, y)

    def prepareData(self, feature, fold_id=-1):
        """
        Reads data from train/test folder and transform raw text 
        and string labels into numpy arrays.
        When fold_id >= 0, the training data is split into n folds 
        and the fold_id will be the development data.  
        """
        if fold_id >= 0:
            pairs = self.readInput(self.train_filename)
            num_instance_per_fold = int(len(pairs[0])  / self.num_folds)
            #fold_id in (0,num_folds - 1)
            if fold_id == 0:
                train_pairs = (pairs[0][(fold_id + 1) * num_instance_per_fold:], pairs[1][(fold_id + 1) * num_instance_per_fold:])
            elif fold_id == self.num_folds - 1:
                train_pairs = (pairs[0][0:fold_id * num_instance_per_fold], pairs[1][0:fold_id * num_instance_per_fold])
            else:
                train_pairs = (pairs[0][0:fold_id * num_instance_per_fold] + pairs[0][(fold_id + 1) * num_instance_per_fold:], pairs[1][0:fold_id * num_instance_per_fold] + pairs[1][(fold_id + 1) * num_instance_per_fold:])
            vectorizer = self.makeVocab(train_pairs, feature)
            train = self.stringToNumpy(train_pairs, vectorizer)
            
            if fold_id == 0:
                dev_pairs = (pairs[0][0:(fold_id + 1)* num_instance_per_fold], pairs[1][0:(fold_id + 1)* num_instance_per_fold])
            elif fold_id == self.num_folds - 1:
                dev_pairs = (pairs[0][fold_id * num_instance_per_fold:], pairs[1][fold_id * num_instance_per_fold:])
            else :
                dev_pairs = (pairs[0][fold_id * num_instance_per_fold: (fold_id + 1) * num_instance_per_fold], pairs[1][fold_id * num_instance_per_fold: (fold_id + 1) * num_instance_per_fold])
            dev = self.stringToNumpy(dev_pairs, vectorizer)
            return train, dev
        else:
            pairs = self.readInput(self.train_filename)
            vectorizer = self.makeVocab(pairs, feature)
            self.vectorizer = vectorizer
            train = self.stringToNumpy(pairs, vectorizer)
            pairs = self.readInput(self.test_filename)
            test = self.stringToNumpy(pairs, vectorizer)
            return train, test



    def makeVocab(self, pairs, feature):
        """
        Convert a collection of text documents to vectorizer, which is a matrix of 
        TF-IDF features or token counts. 
        The vectorizer is used as a transformer from text to vector form.
        """
        vectorizer = None
        if feature == 'tfidf':
            vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             ngram_range=(1, 2),  \
                             binary = True, \
                             #sublinear_tf = True, \
                             #max_features = 20000, \
                             #max_df = 0.1, \
                             )
        elif feature == 'tf':
            vectorizer =  CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             ngram_range=(1, 2),  \
                             binary = True, \
                             #max_features = int(sys.argv[2])
                             #norm = 'l1', \
                             #min_df= 0.01  ) 
                             )
        vectorizer.fit_transform(pairs[0])

        return vectorizer

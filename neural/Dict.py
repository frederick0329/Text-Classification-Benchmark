class Dict:
    def __init__(self):
        self.word2index = {'<pad>':0, '<unk>':1}
        self.word2count = {'<pad>':0, '<unk>':0}
        self.index2word = ['<pad>', '<unk>']
        self.n_words = 2
      
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            if word not in self.word2count:
                self.word2count[word] = 0
            self.word2count[word] += 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def removeLowFreqWords(self, threshold):
        word2index = {'<pad>':0, '<unk>':1}
        word2count = {'<pad>':0, '<unk>':0}
        index2word = ['<pad>', '<unk>']
        n_words = 2
        for word in self.word2count:
            if self.word2count[word] >= threshold:
                word2index[word] = n_words
                word2count[word] = self.word2count[word]
                index2word.append(word)
                n_words += 1
        self.word2index = word2index
        self.word2count = word2count
        self.index2word = index2word
        self.n_words = n_words
        assert(len(index2word) == n_words)
                           
    

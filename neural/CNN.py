import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchwordemb

class CNN(nn.Module):
    def __init__(self, index2word, emb_size,  num_categories, num_kernels, kernels, pretrained_path):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(index2word), emb_size, padding_idx=0)
        self.dropout = nn.Dropout(p=0.3)
        self.index2word = index2word
        self.emb_size = emb_size
        if len(pretrained_path) != 0:
            self.loadPretrained(pretrained_path)
        Ci = 1
        Co = num_kernels
        Ks = [int(i) for i in kernels.split(',')]
        self.convs = [nn.Conv2d(Ci, Co, (K, self.emb_size), padding=(int(K/2), 0)) for K in Ks] 
        self.linear = nn.Linear(len(Ks)*Co, num_categories, bias=True)
  
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # just get the last hidden state
        x = self.linear(x)
        return x

    def loadPretrained(self, pretrained_path):
        count = 0
        vocab, vec = torchwordemb.load_word2vec_bin(pretrained_path)
        for i, word in enumerate(self.index2word):
            if word in vocab:
                count += 1
                self.embedding.weight.data[i,:].copy_(vec[vocab[word]])
            else:
                self.embedding.weight.data[i,:].copy_(torch.FloatTensor(np.random.uniform(-0.25,0.25,300)))

        print("Loaded " + str(count) + " pretrained vectors")

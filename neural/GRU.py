import torch
import torch.nn as nn
from torch.autograd import Variable
import torchwordemb

class GRU(nn.Module):
    def __init__(self, index2word, emb_size, hidden_size, num_categories, bidirectional, pretrained_path):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.index2word = index2word
        self.embedding = nn.Embedding(len(index2word), emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.3)
        self.bidirectional = bidirectional
        if len(pretrained_path) != 0:
            self.loadPretrained(pretrained_path)
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, num_categories) # input dim is 64*2 because its bidirectional
        else:
            self.linear = nn.Linear(hidden_size, num_categories) # input dim is 64*2 because its bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        if self.bidirectional:
            x = x.mean(1)
        x = self.dropout(x[:,-1,:].squeeze(dim=1)) # just get the last hidden state
        x = self.linear(x)
        return x, h

    def init_hidden(self, batch_size):
        if self.bidirectional:
          return Variable(torch.zeros(2, batch_size, self.hidden_size))
        return Variable(torch.zeros(1, batch_size, self.hidden_size))

    def loadPretrained(self, pretrained_path):
        count = 0
        vocab, vec = torchwordemb.load_word2vec_bin(pretrained_path)
        for i, word in enumerate(self.index2word):
            if word in vocab:
                count += 1
                self.embedding.weight.data[i,:].copy_ = vec[vocab[word]]
            else:
                self.embedding.weight.data[i,:].copy_(np.random.uniform(-0.25,0.25,300))
        print("Loaded " + str(count) + " pretrained vectors")

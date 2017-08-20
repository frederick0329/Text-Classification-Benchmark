import numpy as np
import random

class Data:
    def __init__(self, pairs, max_batch_size):
      self.batches = [] 
      self.max_batch_size = max_batch_size
      self.num_instances = len(pairs)
      self.createBatches(pairs)

    def getBatch(self, idx):
        return self.batches[idx]

    def getNumBatches(self):
        return len(self.batches)

    def shuffleBatches(self):
        random.shuffle(self.batches)      
    
    def createBatches(self, pairs):
        pairs.sort(key=lambda x: -len(x[0]))
        begin_length = 0
        for i, pair in enumerate(pairs):
            if i == 0:
                cur_batch_x = []
                cur_batch_y = []
                cur_batch_x.append(pair[0])
                cur_batch_y.append(pair[1])
                begin_length = len(pairs[0][0])
                continue
            if begin_length - 5 < len(pair[0]) and len(cur_batch_x) < self.max_batch_size:
                #padding
                tmp = []
                for j in range(begin_length - len(pair[0])):
                    tmp.append(0)
                tmp = tmp + pair[0]
                cur_batch_x.append(tmp)
                cur_batch_y.append(pair[1])
            else:
                self.batches.append((np.array(cur_batch_x), np.array(cur_batch_y)))
                cur_batch_x = []
                cur_batch_y = []
                begin_length = len(pair[0])
                cur_batch_x.append(pair[0])         
                cur_batch_y.append(pair[1])         
    

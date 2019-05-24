
import numpy as np
from math import floor

class Dataset(object):
    def __init__(self, max_memory):
        self.X = []
        self.y = []
        self.max_memory = max_memory
    
    def add(self, x, y):
        if len(self.X) == self.max_memory:
            self.X = self.X[1:]
            self.y = self.y[1:]

        self.X.append(x)
        self.y.append(y)

    def reset(self):
        self.X = []
        self.y = []

    def get_batch(self, batch_size=32, n_epochs=1):
        n_batches = floor(len(self.X)/batch_size) 

        indices = np.random.permutation(len(self.X))
        X = np.array(self.X)[indices]
        y = np.array(self.y)[indices]

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                X_batch = X[batch_size*batch:batch_size*(batch+1)]
                y_batch = y[batch_size*batch:batch_size*(batch+1)]

                yield X_batch, y_batch

